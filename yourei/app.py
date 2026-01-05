#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import httpx
from textual import on
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Footer, Header, Input, Static, TextArea

from textual.reactive import reactive


# ----------------------------
# Ollama client
# ----------------------------


class OllamaError(RuntimeError):
    pass


async def ollama_generate(
    *,
    base_url: str,
    model: str,
    prompt: str,
    system: str,
    timeout_s: float = 120.0,
) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.4},
    }
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        try:
            r = await client.post(url, json=payload)
        except httpx.RequestError as e:
            raise OllamaError(f"Could not connect to Ollama at {base_url}: {e}") from e

    if r.status_code != 200:
        raise OllamaError(f"Ollama error {r.status_code}: {r.text}")

    data = r.json()
    resp = data.get("response")
    if not isinstance(resp, str) or not resp.strip():
        raise OllamaError(f"Unexpected Ollama response: {data}")
    return resp.strip()


def build_system_prompt() -> str:
    return (
        "You are a language tutor. Be concise and helpful. "
        "Return ONLY valid JSON, no markdown, no code fences, no commentary."
    )


def build_user_prompt(
    term: str, source_lang: str, target_lang: str, n_sentences: int
) -> str:
    return f"""
Create a small learning card for the word/phrase.

Constraints:
- Source language: {source_lang}
- Target language: {target_lang}
- Term: "{term}"
- Provide a natural, context-appropriate translation (not a dictionary list).
- Provide {n_sentences} short example sentences in {source_lang} that use the term naturally.
- For each example sentence, provide its translation in {target_lang}.
- Keep sentences short and realistic.

Return JSON with this exact schema:
{{
  "term": string,
  "translation": string,
  "examples": [
    {{
      "source": string,
      "target": string
    }}
  ]
}}
""".strip()


def parse_card_json(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return JSON.")
    chunk = raw[start : end + 1]
    return json.loads(chunk)


# ----------------------------
# SRS
# ----------------------------


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def today_ymd() -> str:
    return dt.date.today().isoformat()


@dataclass
class SrsCard:
    id: int
    front: str
    source_lang: str
    target_lang: str
    suspended: int

    due: str
    interval_days: int
    reps: int
    ease: float
    lapses: int
    last_review: Optional[str]


def ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY,
            front TEXT NOT NULL,
            source_lang TEXT NOT NULL,
            target_lang TEXT NOT NULL,
            suspended INTEGER NOT NULL DEFAULT 0,

            due TEXT NOT NULL,
            interval_days INTEGER NOT NULL DEFAULT 0,
            reps INTEGER NOT NULL DEFAULT 0,
            ease REAL NOT NULL DEFAULT 2.5,
            lapses INTEGER NOT NULL DEFAULT 0,
            last_review TEXT,

            created_at TEXT NOT NULL,
            llm_translation TEXT,
            llm_examples_json TEXT,
            llm_model TEXT,
            llm_updated_at TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS card_tags (
            card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
            tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
            PRIMARY KEY(card_id, tag_id)
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_card_tags_tag_id
        ON card_tags(tag_id);
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cards_unique
        ON cards(front, source_lang, target_lang);
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY,
            card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
            ts TEXT NOT NULL,
            grade INTEGER NOT NULL,
            prev_due TEXT NOT NULL,
            new_due TEXT NOT NULL,
            prev_interval_days INTEGER NOT NULL,
            new_interval_days INTEGER NOT NULL,
            prev_ease REAL NOT NULL,
            new_ease REAL NOT NULL
        );
        """
    )
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(cards)").fetchall()
    }
    if "llm_translation" not in existing_cols:
        conn.execute("ALTER TABLE cards ADD COLUMN llm_translation TEXT")
    if "llm_examples_json" not in existing_cols:
        conn.execute("ALTER TABLE cards ADD COLUMN llm_examples_json TEXT")
    if "llm_updated_at" not in existing_cols:
        conn.execute("ALTER TABLE cards ADD COLUMN llm_updated_at TEXT")
    if "llm_model" not in existing_cols:
        conn.execute("ALTER TABLE cards ADD COLUMN llm_model TEXT")
    conn.commit()


def add_or_update_card(
    conn: sqlite3.Connection, *, front: str, source_lang: str, target_lang: str
) -> None:
    # New cards start due today. If already exists, leave scheduling as-is.
    conn.execute(
        """
        INSERT INTO cards(front, source_lang, target_lang, due, created_at)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(front, source_lang, target_lang) DO NOTHING
        """,
        (front, source_lang, target_lang, today_ymd(), utc_now_iso()),
    )
    conn.commit()


def get_card_id(
    conn: sqlite3.Connection, *, front: str, source_lang: str, target_lang: str
) -> Optional[int]:
    row = conn.execute(
        """
        SELECT id
        FROM cards
        WHERE front=? AND source_lang=? AND target_lang=?
        """,
        (front, source_lang, target_lang),
    ).fetchone()
    if not row:
        return None
    return int(row[0])


def load_llm_from_db(conn: sqlite3.Connection, card_id: int) -> Optional["LlmCard"]:
    row = conn.execute(
        """
        SELECT front, llm_translation, llm_examples_json, llm_model
        FROM cards WHERE id=?
        """,
        (card_id,),
    ).fetchone()
    if not row:
        return None
    term, translation, examples_json, model = row
    if not translation or not examples_json:
        return None
    try:
        examples = json.loads(examples_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(examples, list):
        return None
    return LlmCard(
        term=str(term),
        translation=str(translation),
        examples=examples,
        model=str(model or ""),
    )


def store_llm_in_db(conn: sqlite3.Connection, card_id: int, llm: "LlmCard") -> None:
    conn.execute(
        """
        UPDATE cards
        SET llm_translation=?, llm_examples_json=?, llm_model=?, llm_updated_at=?
        WHERE id=?
        """,
        (
            llm.translation,
            json.dumps(llm.examples, ensure_ascii=True),
            llm.model,
            utc_now_iso(),
            card_id,
        ),
    )
    conn.commit()


def clear_llm_in_db(conn: sqlite3.Connection, card_id: int) -> None:
    conn.execute(
        """
        UPDATE cards
        SET llm_translation=NULL, llm_examples_json=NULL, llm_model=NULL, llm_updated_at=NULL
        WHERE id=?
        """,
        (card_id,),
    )
    conn.commit()


def iter_words_file(path: Path) -> Iterable[Tuple[str, List[str]]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        if "|" in t:
            term_raw, tags_raw = t.split("|", 1)
        elif "\t" in t:
            term_raw, tags_raw = t.split("\t", 1)
        else:
            yield t, []
            continue
        term = term_raw.strip()
        if not term:
            continue
        tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]
        yield term, tags


def load_due_queue(
    conn: sqlite3.Connection,
    *,
    source_lang: str,
    target_lang: str,
    limit: int = 500,
    tag: Optional[str] = None,
    only_today: bool = False,
) -> List[int]:
    if tag:
        if only_today:
            rows = conn.execute(
                """
                SELECT cards.id
                FROM cards
                JOIN card_tags ON card_tags.card_id = cards.id
                JOIN tags ON tags.id = card_tags.tag_id
                WHERE cards.source_lang=? AND cards.target_lang=? AND cards.suspended=0
                  AND tags.name=? AND cards.due=?
                ORDER BY cards.due ASC, cards.id ASC
                LIMIT ?
                """,
                (source_lang, target_lang, tag, today_ymd(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT cards.id
                FROM cards
                JOIN card_tags ON card_tags.card_id = cards.id
                JOIN tags ON tags.id = card_tags.tag_id
                WHERE cards.source_lang=? AND cards.target_lang=? AND cards.suspended=0
                  AND tags.name=?
                ORDER BY
                  CASE WHEN cards.due <= ? THEN 0 ELSE 1 END,
                  cards.due ASC,
                  cards.id ASC
                LIMIT ?
                """,
                (source_lang, target_lang, tag, today_ymd(), limit),
            ).fetchall()
    else:
        if only_today:
            rows = conn.execute(
                """
                SELECT id
                FROM cards
                WHERE source_lang=? AND target_lang=? AND suspended=0 AND due=?
                ORDER BY due ASC, id ASC
                LIMIT ?
                """,
                (source_lang, target_lang, today_ymd(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id
                FROM cards
                WHERE source_lang=? AND target_lang=? AND suspended=0
                ORDER BY
                  CASE WHEN due <= ? THEN 0 ELSE 1 END,
                  due ASC,
                  id ASC
                LIMIT ?
                """,
                (source_lang, target_lang, today_ymd(), limit),
            ).fetchall()
    return [r[0] for r in rows]


def get_card(conn: sqlite3.Connection, card_id: int) -> SrsCard:
    row = conn.execute(
        """
        SELECT id, front, source_lang, target_lang, suspended,
               due, interval_days, reps, ease, lapses, last_review
        FROM cards WHERE id=?
        """,
        (card_id,),
    ).fetchone()
    if not row:
        raise KeyError(f"Card id {card_id} not found")
    return SrsCard(*row)


def sm2_schedule(card: SrsCard, grade: int) -> Tuple[str, int, int, float, int]:
    grade = int(grade)
    if grade < 0 or grade > 5:
        raise ValueError("grade must be 0..5")

    ease = float(card.ease)
    reps = int(card.reps)
    interval = int(card.interval_days)
    lapses = int(card.lapses)

    q = grade
    ease = ease + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    if ease < 1.3:
        ease = 1.3

    if grade < 3:
        reps = 0
        interval = 0
        lapses += 1
        new_due = dt.date.today()
    else:
        reps += 1
        if reps == 1:
            interval = 1
        elif reps == 2:
            interval = 6
        else:
            interval = max(1, int(round(interval * ease)))
        new_due = dt.date.today() + dt.timedelta(days=interval)

    return (new_due.isoformat(), interval, reps, ease, lapses)


def apply_review(conn: sqlite3.Connection, card: SrsCard, grade: int) -> SrsCard:
    prev_due = card.due
    prev_interval = card.interval_days
    prev_ease = card.ease

    new_due, new_interval, new_reps, new_ease, new_lapses = sm2_schedule(card, grade)

    conn.execute(
        """
        UPDATE cards
        SET due=?, interval_days=?, reps=?, ease=?, lapses=?, last_review=?
        WHERE id=?
        """,
        (new_due, new_interval, new_reps, new_ease, new_lapses, utc_now_iso(), card.id),
    )
    conn.execute(
        """
        INSERT INTO reviews(card_id, ts, grade, prev_due, new_due,
                            prev_interval_days, new_interval_days, prev_ease, new_ease)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            card.id,
            utc_now_iso(),
            int(grade),
            prev_due,
            new_due,
            int(prev_interval),
            int(new_interval),
            float(prev_ease),
            float(new_ease),
        ),
    )
    conn.commit()
    return get_card(conn, card.id)


def stats(
    conn: sqlite3.Connection,
    *,
    source_lang: str,
    target_lang: str,
    tag: Optional[str] = None,
) -> Tuple[int, int, int]:
    if tag:
        total = conn.execute(
            """
            SELECT COUNT(*)
            FROM cards
            JOIN card_tags ON card_tags.card_id = cards.id
            JOIN tags ON tags.id = card_tags.tag_id
            WHERE cards.source_lang=? AND cards.target_lang=? AND tags.name=?
            """,
            (source_lang, target_lang, tag),
        ).fetchone()[0]
        due_now = conn.execute(
            """
            SELECT COUNT(*)
            FROM cards
            JOIN card_tags ON card_tags.card_id = cards.id
            JOIN tags ON tags.id = card_tags.tag_id
            WHERE cards.source_lang=? AND cards.target_lang=? AND cards.suspended=0
              AND cards.due <= ? AND tags.name=?
            """,
            (source_lang, target_lang, today_ymd(), tag),
        ).fetchone()[0]
        suspended = conn.execute(
            """
            SELECT COUNT(*)
            FROM cards
            JOIN card_tags ON card_tags.card_id = cards.id
            JOIN tags ON tags.id = card_tags.tag_id
            WHERE cards.source_lang=? AND cards.target_lang=? AND cards.suspended=1
              AND tags.name=?
            """,
            (source_lang, target_lang, tag),
        ).fetchone()[0]
    else:
        total = conn.execute(
            "SELECT COUNT(*) FROM cards WHERE source_lang=? AND target_lang=?",
            (source_lang, target_lang),
        ).fetchone()[0]
        due_now = conn.execute(
            "SELECT COUNT(*) FROM cards WHERE source_lang=? AND target_lang=? AND suspended=0 AND due <= ?",
            (source_lang, target_lang, today_ymd()),
        ).fetchone()[0]
        suspended = conn.execute(
            "SELECT COUNT(*) FROM cards WHERE source_lang=? AND target_lang=? AND suspended=1",
            (source_lang, target_lang),
        ).fetchone()[0]
    return int(total), int(due_now), int(suspended)


def _get_or_create_tag_id(conn: sqlite3.Connection, tag: str) -> int:
    clean = tag.strip()
    if not clean:
        raise ValueError("Tag cannot be empty.")
    conn.execute(
        """
        INSERT INTO tags(name)
        VALUES(?)
        ON CONFLICT(name) DO NOTHING
        """,
        (clean,),
    )
    row = conn.execute("SELECT id FROM tags WHERE name=?", (clean,)).fetchone()
    if not row:
        raise RuntimeError("Failed to create tag.")
    conn.commit()
    return int(row[0])


def add_tag_to_card(conn: sqlite3.Connection, card_id: int, tag: str) -> None:
    tag_id = _get_or_create_tag_id(conn, tag)
    conn.execute(
        """
        INSERT OR IGNORE INTO card_tags(card_id, tag_id)
        VALUES(?, ?)
        """,
        (card_id, tag_id),
    )
    conn.commit()


def remove_tag_from_card(conn: sqlite3.Connection, card_id: int, tag: str) -> None:
    row = conn.execute("SELECT id FROM tags WHERE name=?", (tag.strip(),)).fetchone()
    if not row:
        return
    conn.execute(
        "DELETE FROM card_tags WHERE card_id=? AND tag_id=?",
        (card_id, int(row[0])),
    )
    conn.commit()


def list_card_tags(conn: sqlite3.Connection, card_id: int) -> List[str]:
    rows = conn.execute(
        """
        SELECT tags.name
        FROM tags
        JOIN card_tags ON card_tags.tag_id = tags.id
        WHERE card_tags.card_id=?
        ORDER BY tags.name COLLATE NOCASE
        """,
        (card_id,),
    ).fetchall()
    return [str(r[0]) for r in rows]


def list_tags_for_deck(
    conn: sqlite3.Connection, *, source_lang: str, target_lang: str
) -> List[Tuple[str, int]]:
    rows = conn.execute(
        """
        SELECT tags.name, COUNT(card_tags.card_id)
        FROM tags
        JOIN card_tags ON card_tags.tag_id = tags.id
        JOIN cards ON cards.id = card_tags.card_id
        WHERE cards.source_lang=? AND cards.target_lang=?
        GROUP BY tags.name
        ORDER BY tags.name COLLATE NOCASE
        """,
        (source_lang, target_lang),
    ).fetchall()
    return [(str(r[0]), int(r[1])) for r in rows]


def list_decks(conn: sqlite3.Connection) -> List[Tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT source_lang, target_lang
        FROM cards
        GROUP BY source_lang, target_lang
        ORDER BY source_lang COLLATE NOCASE, target_lang COLLATE NOCASE
        """
    ).fetchall()
    return [(str(r[0]), str(r[1])) for r in rows]


def suspend_card(conn: sqlite3.Connection, card_id: int, suspend: bool = True) -> None:
    conn.execute(
        "UPDATE cards SET suspended=? WHERE id=?", (1 if suspend else 0, card_id)
    )
    conn.commit()


def delete_card(conn: sqlite3.Connection, card_id: int) -> None:
    conn.execute("DELETE FROM cards WHERE id=?", (card_id,))
    conn.commit()


def delete_card_by_term(
    conn: sqlite3.Connection, *, front: str, source_lang: str, target_lang: str
) -> int:
    cur = conn.execute(
        """
        DELETE FROM cards
        WHERE front=? AND source_lang=? AND target_lang=?
        """,
        (front, source_lang, target_lang),
    )
    conn.commit()
    return int(cur.rowcount or 0)


def update_card_due(conn: sqlite3.Connection, card_id: int, due: str) -> None:
    conn.execute("UPDATE cards SET due=? WHERE id=?", (due, card_id))
    conn.commit()


def toggle_card_suspended(conn: sqlite3.Connection, card_id: int) -> int:
    row = conn.execute("SELECT suspended FROM cards WHERE id=?", (card_id,)).fetchone()
    if not row:
        raise KeyError(f"Card id {card_id} not found")
    new_value = 0 if int(row[0]) else 1
    conn.execute("UPDATE cards SET suspended=? WHERE id=?", (new_value, card_id))
    conn.commit()
    return new_value


def search_cards(
    conn: sqlite3.Connection,
    *,
    source_lang: str,
    target_lang: str,
    query: str,
    tag: Optional[str] = None,
) -> List[Tuple[int, str, int, str]]:
    q = query.strip()
    params: Tuple[object, ...]
    if q:
        like = f"%{q}%"
        if tag:
            sql = """
                SELECT cards.id, cards.front, cards.suspended, cards.due
                FROM cards
                JOIN card_tags ON card_tags.card_id = cards.id
                JOIN tags ON tags.id = card_tags.tag_id
                WHERE cards.source_lang=? AND cards.target_lang=? AND cards.front LIKE ?
                  AND tags.name=?
                ORDER BY cards.front COLLATE NOCASE
            """
            params = (source_lang, target_lang, like, tag)
        else:
            sql = """
                SELECT id, front, suspended, due
                FROM cards
                WHERE source_lang=? AND target_lang=? AND front LIKE ?
                ORDER BY front COLLATE NOCASE
            """
            params = (source_lang, target_lang, like)
    else:
        if tag:
            sql = """
                SELECT cards.id, cards.front, cards.suspended, cards.due
                FROM cards
                JOIN card_tags ON card_tags.card_id = cards.id
                JOIN tags ON tags.id = card_tags.tag_id
                WHERE cards.source_lang=? AND cards.target_lang=? AND tags.name=?
                ORDER BY cards.front COLLATE NOCASE
            """
            params = (source_lang, target_lang, tag)
        else:
            sql = """
                SELECT id, front, suspended, due
                FROM cards
                WHERE source_lang=? AND target_lang=?
                ORDER BY front COLLATE NOCASE
            """
            params = (source_lang, target_lang)
    rows = conn.execute(sql, params).fetchall()
    return [(int(r[0]), str(r[1]), int(r[2]), str(r[3])) for r in rows]


# ----------------------------
# UI Formatting
# ----------------------------


@dataclass
class LlmCard:
    term: str
    translation: str
    examples: List[dict]
    model: str


def format_right_panel(
    *,
    llm: Optional[LlmCard],
    source_lang: str,
    target_lang: str,
    hide_translations: bool = False,
) -> str:
    out: List[str] = []
    out.append(f"{source_lang} → {target_lang}")
    out.append("")
    if llm is None:
        out.append("Generating from local LLM…")
        return "\n".join(out) + "\n"

    if hide_translations:
        out.append("Review mode: translations hidden (press Space to reveal).")
        out.append("")

    model_name = llm.model or "unknown"
    out.append(f"LLM translation (model: {model_name}):")
    if hide_translations:
        out.append("- (hidden)")
    else:
        out.append(f"- {llm.translation}")
    out.append("")
    out.append("LLM examples:")
    for i, ex in enumerate(llm.examples, 1):
        if hide_translations:
            out.append(f"{i}. (hidden)")
            out.append("   → (hidden)")
        else:
            s = (ex.get("source") or "").strip()
            t = (ex.get("target") or "").strip()
            out.append(f"{i}. {s}")
            out.append(f"   → {t}")
    return "\n".join(out).rstrip() + "\n"


def format_deck_results(rows: List[Tuple[int, str, int, str]], query: str) -> str:
    if not rows:
        return f"No results for: {query!r}\n"
    out: List[str] = []
    out.append(f"Results: {len(rows)}")
    out.append("")
    for cid, front, suspended, due in rows:
        status = "S" if suspended else "A"
        out.append(f"[{status}] {front}  (id={cid}, due={due})")
    return "\n".join(out) + "\n"


def format_deck_list(rows: List[Tuple[str, str, int, int, int]]) -> str:
    if not rows:
        return "No decks found.\n"
    out: List[str] = []
    out.append(f"Decks: {len(rows)}")
    out.append("")
    for i, (source, target, total, due_now, suspended) in enumerate(rows, 1):
        out.append(
            f"{i}) {source} → {target} "
            f"(total={total} due={due_now} suspended={suspended})"
        )
    return "\n".join(out) + "\n"


def format_tag_list(rows: List[Tuple[str, int]], current: Optional[str]) -> str:
    if not rows:
        return "No tags found for this deck.\n"
    out: List[str] = []
    out.append(f"Tags: {len(rows)}")
    if current:
        out.append(f"Current filter: {current}")
    out.append("")
    for i, (name, count) in enumerate(rows, 1):
        out.append(f"{i}) {name} (count={count})")
    return "\n".join(out) + "\n"


def format_card_view(
    card: SrsCard,
    llm: Optional[LlmCard],
    source_lang: str,
    target_lang: str,
    tags: List[str],
) -> str:
    out: List[str] = []
    out.append(f"Card id: {card.id}")
    out.append(f"Term: {card.front}")
    out.append(f"Due: {card.due}")
    out.append(f"Suspended: {'yes' if card.suspended else 'no'}")
    out.append(f"Tags: {', '.join(tags) if tags else '-'}")
    out.append("")
    if llm is None:
        out.append("No stored LLM translation/examples for this card.")
        return "\n".join(out) + "\n"
    out.append(f"{source_lang} → {target_lang}")
    out.append("")
    model_name = llm.model or "unknown"
    out.append(f"LLM translation (model: {model_name}):")
    out.append(f"- {llm.translation}")
    out.append("")
    out.append("LLM examples:")
    for i, ex in enumerate(llm.examples, 1):
        s = (ex.get("source") or "").strip()
        t = (ex.get("target") or "").strip()
        out.append(f"{i}. {s}")
        out.append(f"   → {t}")
    return "\n".join(out).rstrip() + "\n"


# ----------------------------
# Modal: Add new word
# ----------------------------
class AddWordModal(ModalScreen[str | None]):
    CSS = """
    AddWordModal {
        align: center middle;
    }
    #dialog {
        width: 70%;
        border: round $secondary;
        padding: 1 2;
    }
    #buttons {
        height: auto;
        padding-top: 1;
    }
    """

    def __init__(
        self, reverse: bool = False, initial_term: Optional[str] = None
    ) -> None:
        super().__init__()
        self.reverse = reverse
        self.initial_term = (initial_term or "").strip()

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            if self.reverse:
                yield Static("Add a new word/phrase to the reverse deck:")
            else:
                yield Static("Add a new word/phrase (will be due today):")
            yield Input(
                placeholder="e.g. nevertheless / to figure out / prendre en compte",
                id="term_input",
            )
            with Horizontal(id="buttons"):
                yield Button("Add", id="add", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        term_input = self.query_one("#term_input", Input)
        if self.initial_term:
            term_input.value = self.initial_term
        term_input.focus()

    def _finish(self, term: str | None) -> None:
        if term:
            self.app.post_message(AddWord(term, reverse=self.reverse))
        self.dismiss(None)

    @on(Button.Pressed)
    def _on_button(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self._finish(None)
            return
        if event.button.id == "add":
            term = self.query_one("#term_input", Input).value.strip()
            self._finish(term if term else None)

    @on(Input.Submitted)
    def _on_submit(self, event: Input.Submitted) -> None:
        term = event.value.strip()
        self._finish(term if term else None)


class PopulateTranslationsModal(ModalScreen[None]):
    CSS = """
    PopulateTranslationsModal {
        align: center middle;
    }
    #dialog {
        width: 70%;
        border: round $secondary;
        padding: 1 2;
    }
    #buttons {
        height: auto;
        padding-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Populate translations for this deck?")
            yield Checkbox(
                "Regenerate all translations using the current model",
                id="regen_all",
            )
            with Horizontal(id="buttons"):
                yield Button("Confirm", id="confirm", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#confirm", Button).focus()

    def _finish(self, confirmed: bool) -> None:
        if confirmed:
            regen_all = self.query_one("#regen_all", Checkbox).value
            self.app.post_message(PopulateTranslations(regen_all))
        self.dismiss(None)

    @on(Button.Pressed)
    def _on_button(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self._finish(False)
            return
        if event.button.id == "confirm":
            self._finish(True)


# ----------------------------
# Modal: Search deck
# ----------------------------


class DeckSearchModal(ModalScreen[None]):
    CSS = """
    DeckSearchModal {
        align: center middle;
    }
    #dialog {
        width: 90%;
        height: 90%;
        border: round $secondary;
        padding: 1 2;
    }
    #search_input {
        height: auto;
        margin-bottom: 1;
    }
    #results {
        height: 1fr;
    }
    #controls {
        height: auto;
        margin-top: 1;
    }
    #tag_controls {
        height: auto;
        margin-top: 1;
    }
    #card_id {
        width: 13;
    }
    #due_date {
        width: 22;
    }
    #toggle_suspend {
        width: auto;
    }
    #tag_input {
        width: 18;
    }
    #remove_tag {
        width: auto;
    }
    #status {
        height: auto;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        source_lang: str,
        target_lang: str,
        tag: Optional[str],
    ) -> None:
        super().__init__()
        self.conn = conn
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.tag = tag

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Search deck (type to filter, Esc to close):")
            yield Input(placeholder="Search term", id="search_input")
            yield TextArea("", id="results", read_only=True)
            with Horizontal(id="controls"):
                yield Input(placeholder="Card id", id="card_id")
                yield Input(placeholder="Due date (YYYY-MM-DD)", id="due_date")
                yield Button("Set due", id="set_due", variant="primary")
                yield Button("View card", id="view_card")
                yield Button("Toggle suspend", id="toggle_suspend")
            with Horizontal(id="tag_controls"):
                yield Input(placeholder="Tag", id="tag_input")
                yield Button("Add tag", id="add_tag")
                yield Button("Remove tag", id="remove_tag")
            yield Static("", id="status")

    def on_mount(self) -> None:
        self.query_one("#search_input", Input).focus()
        self._refresh("")

    def _refresh(self, query: str) -> None:
        rows = search_cards(
            self.conn,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            query=query,
            tag=self.tag,
        )
        text = format_deck_results(rows, query)
        self.query_one("#results", TextArea).load_text(text)
        self.query_one("#status", Static).update("")
        self._maybe_fill_card_id_from_results()

    @on(Input.Changed, "#search_input")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._refresh(event.value)

    @on(Input.Submitted, "#search_input")
    def _on_search_submitted(self, event: Input.Submitted) -> None:
        self._refresh(event.value)

    def _current_results_line(self) -> str:
        results = self.query_one("#results", TextArea)
        text = results.text or ""
        lines = text.splitlines()
        if not lines:
            return ""
        cursor = results.cursor_location
        if isinstance(cursor, tuple):
            row = cursor[0]
        else:
            row = getattr(cursor, "row", 0)
        if row < 0 or row >= len(lines):
            return ""
        return lines[row]

    def _maybe_fill_card_id_from_results(self) -> None:
        card_input = self.query_one("#card_id", Input)
        if card_input.has_focus:
            return
        line = self._current_results_line()
        match = re.search(r"id=(\d+)", line)
        if match:
            card_input.value = match.group(1)

    def _get_selected_card_id(self) -> Optional[int]:
        card_raw = self.query_one("#card_id", Input).value.strip()
        if not card_raw:
            self._maybe_fill_card_id_from_results()
            card_raw = self.query_one("#card_id", Input).value.strip()
        if not card_raw:
            return None
        try:
            return int(card_raw)
        except ValueError:
            return None

    def _set_due(self) -> None:
        card_raw = self.query_one("#card_id", Input).value.strip()
        due_raw = self.query_one("#due_date", Input).value.strip()
        status = self.query_one("#status", Static)
        if not card_raw or not due_raw:
            if not card_raw:
                self._maybe_fill_card_id_from_results()
                card_raw = self.query_one("#card_id", Input).value.strip()
            status.update("Card id and due date are required.")
            return
        try:
            card_id = int(card_raw)
        except ValueError:
            status.update("Card id must be an integer.")
            return
        try:
            due = dt.date.fromisoformat(due_raw).isoformat()
        except ValueError:
            status.update("Due date must be YYYY-MM-DD.")
            return
        update_card_due(self.conn, card_id, due)
        status.update(f"Updated card {card_id} due to {due}.")
        self._refresh(self.query_one("#search_input", Input).value)

    @on(Button.Pressed, "#set_due")
    def _on_set_due(self, event: Button.Pressed) -> None:
        self._set_due()

    @on(Button.Pressed, "#view_card")
    def _on_view_card(self, event: Button.Pressed) -> None:
        status = self.query_one("#status", Static)
        card_id = self._get_selected_card_id()
        if card_id is None:
            status.update("Select a card row or enter a card id.")
            return
        try:
            card = get_card(self.conn, card_id)
        except KeyError:
            status.update(f"Card id {card_id} not found.")
            return
        llm = load_llm_from_db(self.conn, card_id)
        self.app.push_screen(
            CardViewModal(
                card=card,
                llm=llm,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
            )
        )

    @on(Button.Pressed, "#toggle_suspend")
    def _on_toggle_suspend(self, event: Button.Pressed) -> None:
        status = self.query_one("#status", Static)
        card_id = self._get_selected_card_id()
        if card_id is None:
            status.update("Select a card row or enter a card id.")
            return
        try:
            new_value = toggle_card_suspended(self.conn, card_id)
        except KeyError:
            status.update(f"Card id {card_id} not found.")
            return
        status.update("Card suspended." if new_value else "Card active.")
        self._refresh(self.query_one("#search_input", Input).value)

    def _add_tag(self) -> None:
        status = self.query_one("#status", Static)
        card_id = self._get_selected_card_id()
        if card_id is None:
            status.update("Select a card row or enter a card id.")
            return
        tag = self.query_one("#tag_input", Input).value.strip()
        if not tag:
            status.update("Tag cannot be empty.")
            return
        add_tag_to_card(self.conn, card_id, tag)
        status.update(f"Added tag '{tag}' to card {card_id}.")

    @on(Button.Pressed, "#add_tag")
    def _on_add_tag(self, event: Button.Pressed) -> None:
        self._add_tag()

    def _remove_tag(self) -> None:
        status = self.query_one("#status", Static)
        card_id = self._get_selected_card_id()
        if card_id is None:
            status.update("Select a card row or enter a card id.")
            return
        tag = self.query_one("#tag_input", Input).value.strip()
        if not tag:
            status.update("Tag cannot be empty.")
            return
        remove_tag_from_card(self.conn, card_id, tag)
        status.update(f"Removed tag '{tag}' from card {card_id}.")

    @on(Button.Pressed, "#remove_tag")
    def _on_remove_tag(self, event: Button.Pressed) -> None:
        self._remove_tag()

    @on(Input.Submitted, "#due_date")
    def _on_due_submitted(self, event: Input.Submitted) -> None:
        self._set_due()

    @on(Input.Submitted, "#tag_input")
    def _on_tag_submitted(self, event: Input.Submitted) -> None:
        self._add_tag()

    @on(TextArea.SelectionChanged, "#results")
    def _on_results_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        self._maybe_fill_card_id_from_results()

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


class CardViewModal(ModalScreen[None]):
    CSS = """
    CardViewModal {
        align: center middle;
    }
    #dialog {
        width: 90%;
        height: 90%;
        border: round $secondary;
        padding: 1 2;
    }
    #card_view {
        height: 1fr;
    }
    """

    def __init__(
        self,
        *,
        card: SrsCard,
        llm: Optional[LlmCard],
        source_lang: str,
        target_lang: str,
    ) -> None:
        super().__init__()
        self.card = card
        self.llm = llm
        self.source_lang = source_lang
        self.target_lang = target_lang

    @property
    def app(self):
        """Returns the main App instance"""
        return super().app

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Card details (Esc to close):")
            yield TextArea("", id="card_view", read_only=True)

    def on_mount(self) -> None:
        if self.llm is None:
            cached = getattr(self.app, "llm_cache", {}).get(self.card.front)
            if cached is not None:
                self.llm = cached
        self._update_view()

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)
        if event.key == "g":
            self.query_one("#card_view", TextArea).load_text(
                "Generating from local LLM…\n"
            )
            asyncio.create_task(self._generate_llm())

    def _update_view(self) -> None:
        tags = list_card_tags(self.app.conn, self.card.id)
        text = format_card_view(
            self.card,
            self.llm,
            self.source_lang,
            self.target_lang,
            tags,
        )
        self.query_one("#card_view", TextArea).load_text(text)

    async def _generate_llm(self) -> None:
        try:
            system = build_system_prompt()
            prompt = build_user_prompt(
                self.card.front,
                self.source_lang,
                self.target_lang,
                self.app.sentences,
            )
            raw = await ollama_generate(
                base_url=self.app.ollama_url,
                model=self.app.model,
                prompt=prompt,
                system=system,
            )
            data = parse_card_json(raw)
            llm = LlmCard(
                term=str(data.get("term", self.card.front)),
                translation=str(data.get("translation", "")).strip(),
                examples=list(data.get("examples", []))
                if isinstance(data.get("examples", []), list)
                else [],
                model=self.app.model,
            )
            if not llm.translation:
                raise ValueError("Missing translation in JSON.")
            if not llm.examples:
                raise ValueError("Missing examples in JSON.")
            self.llm = llm
            store_llm_in_db(self.app.conn, self.card.id, llm)
            cache = getattr(self.app, "llm_cache", None)
            if isinstance(cache, dict):
                cache[self.card.front] = llm
            if self.is_attached:
                self._update_view()
        except Exception as exc:
            if self.is_attached:
                self.query_one("#card_view", TextArea).load_text(
                    f"Error generating from Ollama.\n\n{exc}\n"
                )


class AddTagModal(ModalScreen[None]):
    CSS = """
    AddTagModal {
        align: center middle;
    }
    #dialog {
        width: 60%;
        border: round $secondary;
        padding: 1 2;
    }
    #buttons {
        height: auto;
        padding-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Add tag to current card:")
            yield Input(placeholder="e.g. verbs / travel / past-tense", id="tag_input")
            with Horizontal(id="buttons"):
                yield Button("Add", id="add", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#tag_input", Input).focus()

    def _finish(self, tag: str | None) -> None:
        if tag:
            self.app.post_message(AddTag(tag))
        self.dismiss(None)

    @on(Button.Pressed)
    def _on_button(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self._finish(None)
            return
        if event.button.id == "add":
            tag = self.query_one("#tag_input", Input).value.strip()
            self._finish(tag if tag else None)

    @on(Input.Submitted)
    def _on_submit(self, event: Input.Submitted) -> None:
        tag = event.value.strip()
        self._finish(tag if tag else None)


class RemoveTagModal(ModalScreen[None]):
    CSS = """
    RemoveTagModal {
        align: center middle;
    }
    #dialog {
        width: 60%;
        border: round $secondary;
        padding: 1 2;
    }
    #buttons {
        height: auto;
        padding-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Remove tag from current card:")
            yield Input(placeholder="Tag", id="tag_input")
            with Horizontal(id="buttons"):
                yield Button("Remove", id="remove", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#tag_input", Input).focus()

    def _finish(self, tag: str | None) -> None:
        if tag:
            self.app.post_message(RemoveTag(tag))
        self.dismiss(None)

    @on(Button.Pressed)
    def _on_button(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self._finish(None)
            return
        if event.button.id == "remove":
            tag = self.query_one("#tag_input", Input).value.strip()
            self._finish(tag if tag else None)

    @on(Input.Submitted)
    def _on_submit(self, event: Input.Submitted) -> None:
        tag = event.value.strip()
        self._finish(tag if tag else None)


class ModelSwitchModal(ModalScreen[None]):
    CSS = """
    ModelSwitchModal {
        align: center middle;
    }
    #dialog {
        width: 60%;
        border: round $secondary;
        padding: 1 2;
    }
    #buttons {
        height: auto;
        padding-top: 1;
    }
    """

    def __init__(self, *, current_model: str) -> None:
        super().__init__()
        self.current_model = current_model

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Switch Ollama model:")
            yield Input(placeholder="e.g. llama3.1", id="model_input")
            with Horizontal(id="buttons"):
                yield Button("Switch", id="switch", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        model_input = self.query_one("#model_input", Input)
        model_input.value = self.current_model
        model_input.focus()

    def _finish(self, model: str | None) -> None:
        if model:
            self.app.post_message(SetModel(model))
        self.dismiss(None)

    @on(Button.Pressed)
    def _on_button(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self._finish(None)
            return
        if event.button.id == "switch":
            model = self.query_one("#model_input", Input).value.strip()
            self._finish(model if model else None)

    @on(Input.Submitted)
    def _on_submit(self, event: Input.Submitted) -> None:
        model = event.value.strip()
        self._finish(model if model else None)


class TagFilterModal(ModalScreen[None]):
    CSS = """
    TagFilterModal {
        align: center middle;
    }
    #dialog {
        width: 80%;
        height: 80%;
        border: round $secondary;
        padding: 1 2;
    }
    #tag_list {
        height: 1fr;
    }
    #controls {
        height: auto;
        margin-top: 1;
    }
    #tag_filter {
        width: 18;
    }
    #status {
        height: auto;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        source_lang: str,
        target_lang: str,
        current_tag: Optional[str],
    ) -> None:
        super().__init__()
        self.conn = conn
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.current_tag = current_tag
        self._rows: List[Tuple[str, int]] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Filter by tag (select a row or type a tag, Esc to close):")
            yield TextArea("", id="tag_list", read_only=True)
            with Horizontal(id="controls"):
                yield Input(placeholder="Tag", id="tag_filter")
                yield Button("Set filter", id="set_filter", variant="primary")
                yield Button("Clear filter", id="clear_filter")
            yield Static("", id="status")

    def on_mount(self) -> None:
        self.query_one("#tag_filter", Input).focus()
        self._refresh()

    def _refresh(self) -> None:
        rows = list_tags_for_deck(
            self.conn,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
        )
        self._rows = rows
        text = format_tag_list(rows, self.current_tag)
        self.query_one("#tag_list", TextArea).load_text(text)
        self.query_one("#status", Static).update("")
        self._maybe_fill_tag_from_list()
        if self.current_tag:
            self.query_one("#tag_filter", Input).value = self.current_tag

    def _current_list_line(self) -> str:
        results = self.query_one("#tag_list", TextArea)
        text = results.text or ""
        lines = text.splitlines()
        if not lines:
            return ""
        cursor = results.cursor_location
        if isinstance(cursor, tuple):
            row = cursor[0]
        else:
            row = getattr(cursor, "row", 0)
        if row < 0 or row >= len(lines):
            return ""
        return lines[row]

    def _maybe_fill_tag_from_list(self) -> None:
        tag_input = self.query_one("#tag_filter", Input)
        if tag_input.has_focus:
            return
        line = self._current_list_line().strip()
        match = re.match(r"\d+\)\s+([^\(]+)", line)
        if match:
            tag_input.value = match.group(1).strip()

    def _set_filter(self) -> None:
        status = self.query_one("#status", Static)
        tag = self.query_one("#tag_filter", Input).value.strip()
        if not tag:
            status.update("Tag cannot be empty.")
            return
        self.app.post_message(SetTagFilter(tag))
        self.dismiss(None)

    def _clear_filter(self) -> None:
        self.app.post_message(SetTagFilter(None))
        self.dismiss(None)

    @on(Button.Pressed, "#set_filter")
    def _on_set_filter(self, event: Button.Pressed) -> None:
        self._set_filter()

    @on(Button.Pressed, "#clear_filter")
    def _on_clear_filter(self, event: Button.Pressed) -> None:
        self._clear_filter()

    @on(Input.Submitted, "#tag_filter")
    def _on_tag_submitted(self, event: Input.Submitted) -> None:
        self._set_filter()

    @on(TextArea.SelectionChanged, "#tag_list")
    def _on_list_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        self._maybe_fill_tag_from_list()

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


class DeckSwitchModal(ModalScreen[None]):
    CSS = """
    DeckSwitchModal {
        align: center middle;
    }
    #dialog {
        width: 80%;
        height: 80%;
        border: round $secondary;
        padding: 1 2;
    }
    #deck_list {
        height: 1fr;
    }
    #controls {
        height: auto;
        margin-top: 1;
    }
    #deck_index {
        width: 8;
    }
    #status {
        height: auto;
        margin-top: 1;
    }
    """

    def __init__(self, *, conn: sqlite3.Connection) -> None:
        super().__init__()
        self.conn = conn
        self._rows: List[Tuple[str, str, int, int, int]] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Switch deck (select a row or enter a number, Esc to close):")
            yield TextArea("", id="deck_list", read_only=True)
            with Horizontal(id="controls"):
                yield Input(placeholder="#", id="deck_index")
                yield Button("Switch", id="switch", variant="primary")
            yield Static("", id="status")

    def on_mount(self) -> None:
        self.query_one("#deck_index", Input).focus()
        self._refresh()

    def _refresh(self) -> None:
        decks = list_decks(self.conn)
        rows: List[Tuple[str, str, int, int, int]] = []
        for source_lang, target_lang in decks:
            total, due_now, suspended = stats(
                self.conn, source_lang=source_lang, target_lang=target_lang
            )
            rows.append((source_lang, target_lang, total, due_now, suspended))
        self._rows = rows
        text = format_deck_list(rows)
        self.query_one("#deck_list", TextArea).load_text(text)
        self.query_one("#status", Static).update("")
        self._maybe_fill_index_from_list()

    def _current_list_line(self) -> str:
        results = self.query_one("#deck_list", TextArea)
        text = results.text or ""
        lines = text.splitlines()
        if not lines:
            return ""
        cursor = results.cursor_location
        if isinstance(cursor, tuple):
            row = cursor[0]
        else:
            row = getattr(cursor, "row", 0)
        if row < 0 or row >= len(lines):
            return ""
        return lines[row]

    def _maybe_fill_index_from_list(self) -> None:
        index_input = self.query_one("#deck_index", Input)
        if index_input.has_focus:
            return
        line = self._current_list_line()
        match = re.match(r"(\d+)\)", line.strip())
        if match:
            index_input.value = match.group(1)

    def _get_selected_deck(self) -> Optional[Tuple[str, str]]:
        raw = self.query_one("#deck_index", Input).value.strip()
        if not raw:
            self._maybe_fill_index_from_list()
            raw = self.query_one("#deck_index", Input).value.strip()
        if not raw:
            return None
        try:
            idx = int(raw)
        except ValueError:
            return None
        if idx < 1 or idx > len(self._rows):
            return None
        source_lang, target_lang, _, _, _ = self._rows[idx - 1]
        return (source_lang, target_lang)

    def _switch(self) -> None:
        status = self.query_one("#status", Static)
        selected = self._get_selected_deck()
        if not selected:
            status.update("Select a deck row or enter a valid number.")
            return
        source_lang, target_lang = selected
        self.app.post_message(SwitchDeck(source_lang, target_lang))
        self.dismiss(None)

    @on(Button.Pressed, "#switch")
    def _on_switch(self, event: Button.Pressed) -> None:
        self._switch()

    @on(Input.Submitted, "#deck_index")
    def _on_index_submitted(self, event: Input.Submitted) -> None:
        self._switch()

    @on(TextArea.SelectionChanged, "#deck_list")
    def _on_list_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        self._maybe_fill_index_from_list()

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


# ----------------------------
# Textual App
# ----------------------------


class LlmReady(Message):
    def __init__(self, term: str, llm: LlmCard) -> None:
        super().__init__()
        self.term = term
        self.llm = llm


class ErrorMsg(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class AddWord(Message):
    def __init__(self, term: str, reverse: bool = False) -> None:
        super().__init__()
        self.term = term
        self.reverse = reverse


class PopulateTranslations(Message):
    def __init__(self, regenerate_all: bool) -> None:
        super().__init__()
        self.regenerate_all = regenerate_all


class SwitchDeck(Message):
    def __init__(self, source_lang: str, target_lang: str) -> None:
        super().__init__()
        self.source_lang = source_lang
        self.target_lang = target_lang


class AddTag(Message):
    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag


class RemoveTag(Message):
    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag


class SetTagFilter(Message):
    def __init__(self, tag: Optional[str]) -> None:
        super().__init__()
        self.tag = tag


class SetModel(Message):
    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model


class LeftPanel(Static):
    term: reactive[str] = reactive("")
    info: reactive[str] = reactive("")

    def render(self) -> str:
        return f"{self.term}\n\n{self.info}".strip()


class Yourei(App):
    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; }
    #left {
        width: 30%;
        border: round $secondary;
        padding: 1 2;
        height: 1fr;
    }
    #right {
        width: 70%;
        border: round $secondary;
        padding: 1 2;
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("0", "grade(0)", "Again"),
        Binding("1", "grade(1)", "Bad"),
        Binding("2", "grade(2)", "Hard"),
        Binding("3", "grade(3)", "OK"),
        Binding("4", "grade(4)", "Good"),
        Binding("5", "grade(5)", "Easy"),
        Binding("g", "regen_llm", "(Re)generate"),
        Binding("u", "populate_translations", "Populate", show=False),
        Binding("n", "skip", "Skip"),
        Binding("b", "prev", "Back"),
        Binding("x", "suspend", "Suspend"),
        Binding("a", "add_word", "Add"),
        Binding("A", "add_word_reverse", "Add reverse"),
        Binding("d", "delete_card", "Delete"),
        Binding("t", "filter_tag", "Filter", show=False),
        Binding("l", "add_tag", "Tag", show=False),
        Binding("r", "remove_tag", "Untag", show=False),
        Binding("s", "search_deck", "Search", show=False),
        Binding("c", "switch_deck", "Switch", show=False),
        Binding("m", "switch_model", "Model"),
        Binding("v", "toggle_review_mode", "Review mode"),
        Binding("space", "reveal_translation", "Reveal", show=False),
        Binding("p", "save_config", "Save config", show=False),
        Binding("y", "toggle_today", "Due today"),
        Binding("q", "quit", "Quit"),
    ]

    def get_system_commands(self, screen) -> Iterable[SystemCommand]:
        for cmd in super().get_system_commands(screen):
            if cmd.title == "Keys":
                continue
            yield cmd
        yield SystemCommand(
            "Switch Deck", "Switch source/target deck", self.action_switch_deck
        )
        yield SystemCommand(
            "Search Deck", "Search within a deck", self.action_search_deck
        )
        yield SystemCommand(
            "Save Config", "Save current settings to config", self.action_save_config
        )
        yield SystemCommand("Filter", "Filter deck by tag", self.action_filter_tag)
        yield SystemCommand(
            "Populate Translations",
            "Update or populate translations with current LLM",
            self.action_populate_translations,
        )

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        source_lang: str,
        target_lang: str,
        model: str,
        ollama_url: str,
        sentences: int,
        theme: Optional[str],
        config_path: Path,
        db_path: Path,
    ) -> None:
        super().__init__()
        if theme:
            self.theme = theme
        self.conn = conn
        self.config_path = config_path
        self.db_path = db_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model = model
        self.ollama_url = ollama_url
        self.sentences = sentences

        self.queue: List[int] = []
        self.pos: int = 0
        self.current: Optional[SrsCard] = None
        self.llm_cache: dict[str, LlmCard] = {}
        self.tag_filter: Optional[str] = None
        self.only_today: bool = False
        self.last_selected_text: Optional[str] = None
        self.review_mode: bool = False
        self.reveal_translation: bool = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main"):
            yield LeftPanel(id="left")
            yield TextArea("", id="right", read_only=True)
        yield Footer(show_command_palette=False)

    def on_mount(self) -> None:
        self._reload_queue()
        self.call_later(self._load_current)

    def _reload_queue(self) -> None:
        self.queue = load_due_queue(
            self.conn,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            tag=self.tag_filter,
            only_today=self.only_today,
        )
        self.pos = 0

    def _list_translation_targets(self, regenerate_all: bool) -> List[Tuple[int, str]]:
        if regenerate_all:
            rows = self.conn.execute(
                """
                SELECT id, front
                FROM cards
                WHERE source_lang=? AND target_lang=?
                ORDER BY id ASC
                """,
                (self.source_lang, self.target_lang),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT id, front
                FROM cards
                WHERE source_lang=? AND target_lang=?
                  AND (
                    llm_translation IS NULL OR llm_translation=''
                    OR llm_examples_json IS NULL OR llm_examples_json=''
                  )
                ORDER BY id ASC
                """,
                (self.source_lang, self.target_lang),
            ).fetchall()
        return [(int(r[0]), str(r[1])) for r in rows]

    async def _populate_translations(self, regenerate_all: bool) -> None:
        targets = self._list_translation_targets(regenerate_all)
        if not targets:
            self._update_left("No translations to generate.")
            self._set_right("No cards need translation updates.\n")
            return

        total = len(targets)
        errors = 0
        for idx, (card_id, term) in enumerate(targets, 1):
            self._update_left(f"Generating translations ({idx}/{total})…")
            self._set_right(
                f"Generating translations from local LLM…\n{idx}/{total}: {term}\n"
            )
            try:
                system = build_system_prompt()
                prompt = build_user_prompt(
                    term,
                    self.source_lang,
                    self.target_lang,
                    self.sentences,
                )
                raw = await ollama_generate(
                    base_url=self.ollama_url,
                    model=self.model,
                    prompt=prompt,
                    system=system,
                )
                data = parse_card_json(raw)
                llm = LlmCard(
                    term=str(data.get("term", term)),
                    translation=str(data.get("translation", "")).strip(),
                    examples=list(data.get("examples", []))
                    if isinstance(data.get("examples", []), list)
                    else [],
                    model=self.model,
                )
                if not llm.translation:
                    raise ValueError("Missing translation in JSON.")
                if not llm.examples:
                    raise ValueError("Missing examples in JSON.")
                store_llm_in_db(self.conn, card_id, llm)
                self.llm_cache[term] = llm
            except Exception as exc:
                errors += 1
                self._set_right(f"Error generating for {term}:\n{exc}\n")
            await asyncio.sleep(0)

        if errors:
            self._update_left(f"Done with {errors} error(s).")
            self._set_right(f"Finished with {errors} error(s).\n")
        else:
            self._update_left("Translations generated.")
            self._set_right("Finished generating translations.\n")
        self._load_current()

    def _set_right(self, text: str) -> None:
        self.query_one("#right", TextArea).load_text(text)

    def _update_left(self, status: str) -> None:
        left = self.query_one("#left", LeftPanel)
        total, due_now, suspended = stats(
            self.conn,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            tag=self.tag_filter,
        )

        if self.current:
            term = self.current.front
            due = self.current.due
            reps = self.current.reps
            itv = self.current.interval_days
            ef = self.current.ease
            lap = self.current.lapses
            tags = list_card_tags(self.conn, self.current.id)
        else:
            term = "(no cards)"
            due = "-"
            reps = itv = lap = 0
            ef = 0.0
            tags = []

        tag_info = ", ".join(tags) if tags else "-"
        filter_info = self.tag_filter if self.tag_filter else "-"
        due_info = "today" if self.only_today else "any"
        review_info = "on" if self.review_mode else "off"
        left.term = term
        left.info = (
            f"Model: {self.model}\n"
            f"Ollama: {self.ollama_url}\n"
            f"\nDeck: {self.source_lang} → {self.target_lang} "
            f"total={total} due={due_now} suspended={suspended}\n"
            f"Filter: {filter_info} • Due: {due_info} • Review: {review_info}\n"
            f"Card: due={due} reps={reps} interval={itv}d ease={ef:.2f} lapses={lap}\n"
            f"Tags: {tag_info}\n"
            f"\nKeys: • 0-5 grade • g regenerate from LLM • n skip • x suspend • a add • d delete "
            f"• u populate translations • b back • l tag • r untag • m change model • v toggle review mode • space reveal • toggle due today  • s search deck • c switch deck • t filter deck "
            f"• p save config • q quit"
            f"\n\nStatus: {status}\n"
        )

    def _load_current(self) -> None:
        if not self.queue:
            self.current = None
            filter_note = f" for tag '{self.tag_filter}'" if self.tag_filter else ""
            self._update_left("No cards loaded. Press 'a' to add one.")
            self._set_right(
                "No cards available"
                f"{filter_note}.\n\nPress 'a' to add a new word/phrase.\n"
            )
            return

        self.pos = max(0, min(self.pos, len(self.queue) - 1))
        cid = self.queue[self.pos]
        self.current = get_card(self.conn, cid)
        if self.review_mode:
            self.reveal_translation = False

        term = self.current.front
        llm = self.llm_cache.get(term)
        if llm is None:
            llm = load_llm_from_db(self.conn, cid)
            if llm is not None:
                self.llm_cache[term] = llm

        self._update_left("Loaded." if llm else "Generating…")
        self._set_right(
            format_right_panel(
                llm=llm,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                hide_translations=self.review_mode and not self.reveal_translation,
            )
        )

        if llm is None:
            asyncio.create_task(self._generate_llm(term, cid))

    async def _generate_llm(self, term: str, card_id: int) -> None:
        try:
            system = build_system_prompt()
            prompt = build_user_prompt(
                term, self.source_lang, self.target_lang, self.sentences
            )
            raw = await ollama_generate(
                base_url=self.ollama_url,
                model=self.model,
                prompt=prompt,
                system=system,
            )
            data = parse_card_json(raw)
            llm = LlmCard(
                term=str(data.get("term", term)),
                translation=str(data.get("translation", "")).strip(),
                examples=list(data.get("examples", []))
                if isinstance(data.get("examples", []), list)
                else [],
                model=self.model,
            )
            if not llm.translation:
                raise ValueError("Missing translation in JSON.")
            if not llm.examples:
                raise ValueError("Missing examples in JSON.")
            self.llm_cache[term] = llm
            store_llm_in_db(self.conn, card_id, llm)
            self.post_message(LlmReady(term, llm))
        except Exception as e:
            self.post_message(ErrorMsg(str(e)))

    def on_llm_ready(self, msg: LlmReady) -> None:
        if not self.current:
            return
        if msg.term == self.current.front:
            self._update_left("Generated.")
            self._set_right(
                format_right_panel(
                    llm=msg.llm,
                    source_lang=self.source_lang,
                    target_lang=self.target_lang,
                    hide_translations=self.review_mode and not self.reveal_translation,
                )
            )

    def on_error_msg(self, msg: ErrorMsg) -> None:
        self._update_left("Error.")
        self._set_right(
            "Error generating from Ollama.\n\n"
            f"{msg.text}\n\n"
            "Tips:\n- Ensure `ollama serve` is running\n- Ensure the model name matches `ollama list`\n"
        )

    # Actions
    def action_toggle_review_mode(self) -> None:
        self.review_mode = not self.review_mode
        if self.review_mode:
            self.reveal_translation = False
        self._load_current()

    def action_reveal_translation(self) -> None:
        if not self.review_mode or not self.current:
            return
        if self.reveal_translation:
            return
        self.reveal_translation = True
        llm = self.llm_cache.get(self.current.front)
        if llm is None:
            llm = load_llm_from_db(self.conn, self.current.id)
            if llm is not None:
                self.llm_cache[self.current.front] = llm
        self._set_right(
            format_right_panel(
                llm=llm,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                hide_translations=False,
            )
        )

    def action_regen_llm(self) -> None:
        if not self.current:
            return
        self.llm_cache.pop(self.current.front, None)
        clear_llm_in_db(self.conn, self.current.id)
        self._load_current()

    def action_populate_translations(self) -> None:
        self.push_screen(PopulateTranslationsModal())

    def action_skip(self) -> None:
        if not self.queue:
            return
        self.pos += 1
        if self.pos >= len(self.queue):
            self._reload_queue()
        self._load_current()

    def action_prev(self) -> None:
        if not self.queue:
            return
        if self.pos > 0:
            self.pos -= 1
        self._load_current()

    def action_suspend(self) -> None:
        if not self.current:
            return
        suspend_card(self.conn, self.current.id, True)
        if self.queue and self.queue[self.pos] == self.current.id:
            self.queue.pop(self.pos)
            if self.pos >= len(self.queue):
                self.pos = 0
        self._load_current()

    def action_delete_card(self) -> None:
        if not self.current:
            return
        delete_card(self.conn, self.current.id)
        self.llm_cache.pop(self.current.front, None)
        if self.queue and self.queue[self.pos] == self.current.id:
            self.queue.pop(self.pos)
            if self.pos >= len(self.queue):
                self.pos = 0
        self._reload_queue()
        self._load_current()

    def action_search_deck(self) -> None:
        self.push_screen(
            DeckSearchModal(
                conn=self.conn,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                tag=self.tag_filter,
            )
        )

    def action_switch_deck(self) -> None:
        self.push_screen(DeckSwitchModal(conn=self.conn))

    def action_switch_model(self) -> None:
        self.push_screen(ModelSwitchModal(current_model=self.model))

    def action_filter_tag(self) -> None:
        self.push_screen(
            TagFilterModal(
                conn=self.conn,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                current_tag=self.tag_filter,
            )
        )

    def action_add_tag(self) -> None:
        if not self.current:
            return
        self.push_screen(AddTagModal())

    def action_remove_tag(self) -> None:
        if not self.current:
            return
        self.push_screen(RemoveTagModal())

    def action_grade(self, grade: int) -> None:
        if not self.current:
            return
        _ = apply_review(self.conn, self.current, grade)
        self._reload_queue()
        self._load_current()

    def action_save_config(self) -> None:
        theme_value = self.theme if isinstance(self.theme, str) else None
        try:
            save_toml_config(
                self.config_path,
                {
                    "db": str(self.db_path),
                    "source": self.source_lang,
                    "target": self.target_lang,
                    "model": self.model,
                    "ollama_url": self.ollama_url,
                    "sentences": self.sentences,
                    "theme": theme_value,
                },
            )
        except Exception as exc:
            self._update_left("Config save failed.")
            self._set_right(f"Failed to save config:\n{exc}\n")
            return
        self._update_left(f"Config saved to {self.config_path}.")

    def action_toggle_today(self) -> None:
        self.only_today = not self.only_today
        self._reload_queue()
        self._load_current()

    def _prefill_term(self) -> Optional[str]:
        selected = (self.last_selected_text or "").strip()
        if not selected:
            return None
        return " ".join(selected.split())

    def action_add_word(self) -> None:
        self.push_screen(AddWordModal(initial_term=self._prefill_term()))

    def action_add_word_reverse(self) -> None:
        self.push_screen(AddWordModal(reverse=True, initial_term=self._prefill_term()))

    @on(TextArea.SelectionChanged, "#right")
    def _on_right_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        selected = event.control.selected_text.strip()
        if selected:
            self.last_selected_text = selected
        else:
            self.last_selected_text = None

    def on_add_word(self, msg: AddWord) -> None:
        term = msg.term.strip()
        if not term:
            return
        source_lang = self.source_lang
        target_lang = self.target_lang
        if msg.reverse:
            source_lang, target_lang = target_lang, source_lang
        add_or_update_card(
            self.conn,
            front=term,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        self.llm_cache.pop(term, None)
        if msg.reverse:
            self._update_left(f"Added {term} to {source_lang} → {target_lang} deck.")
            return
        card_id = get_card_id(
            self.conn,
            front=term,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        self._reload_queue()
        if card_id is not None:
            try:
                self.pos = self.queue.index(card_id)
            except ValueError:
                self.queue.insert(0, card_id)
                self.pos = 0
        self._load_current()

    def on_populate_translations(self, msg: PopulateTranslations) -> None:
        asyncio.create_task(self._populate_translations(msg.regenerate_all))

    def on_switch_deck(self, msg: SwitchDeck) -> None:
        if msg.source_lang == self.source_lang and msg.target_lang == self.target_lang:
            return
        self.source_lang = msg.source_lang
        self.target_lang = msg.target_lang
        self.tag_filter = None
        self.llm_cache.clear()
        self._reload_queue()
        self._load_current()

    def on_add_tag(self, msg: AddTag) -> None:
        if not self.current:
            return
        tag = msg.tag.strip()
        if not tag:
            return
        add_tag_to_card(self.conn, self.current.id, tag)
        self._update_left("Tag added.")

    def on_remove_tag(self, msg: RemoveTag) -> None:
        if not self.current:
            return
        tag = msg.tag.strip()
        if not tag:
            return
        remove_tag_from_card(self.conn, self.current.id, tag)
        self._update_left("Tag removed.")

    def on_set_tag_filter(self, msg: SetTagFilter) -> None:
        self.tag_filter = msg.tag.strip() if msg.tag else None
        self._reload_queue()
        self._load_current()

    def on_set_model(self, msg: SetModel) -> None:
        model = msg.model.strip()
        if not model or model == self.model:
            return
        self.model = model
        self.llm_cache.clear()
        self._update_left("Model updated. Press 'g' to regenerate.")


# ----------------------------
# CLI / main
# ----------------------------


def _load_toml_config(path: Path, section: str) -> dict[str, object]:
    try:
        import tomllib as toml_mod
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - optional dependency for Python < 3.11
        try:
            import tomli as toml_mod  # type: ignore[no-redef]
        except ModuleNotFoundError:
            raise RuntimeError("TOML config requires Python 3.11+ or tomli installed.")

    if not path.exists():
        return {}

    with path.open("rb") as handle:
        data = toml_mod.load(handle)
    if not isinstance(data, dict):
        return {}
    block = data.get(section)
    if isinstance(block, dict):
        return dict(block)
    return data


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _toml_value(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return f'"{_toml_escape(value)}"'
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def save_toml_config(path: Path, settings: dict[str, object]) -> None:
    lines = ["[yourei]"]
    for key in ("db", "source", "target", "model", "ollama_url", "sentences", "theme"):
        value = settings.get(key)
        if value is None:
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _coerce_str(value: object) -> Optional[str]:
    if isinstance(value, str):
        return value
    return None


def _coerce_int(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Local spaced repetition TUI using Ollama (SQLite + SM-2)."
    )
    p.add_argument("--config", type=str, default=None, help="Path to TOML config file.")
    p.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite DB path (default: ~/.yourei/yourei.sqlite3).",
    )
    p.add_argument("--source", type=str, default=None, help="Source language label.")
    p.add_argument("--target", type=str, default=None, help="Target language label.")
    p.add_argument(
        "--model", type=str, default=None, help="Ollama model name (e.g. llama3.1)."
    )
    p.add_argument("--ollama-url", type=str, default=None, help="Ollama base URL.")
    p.add_argument(
        "--sentences",
        type=int,
        default=None,
        help="Number of example sentences to generate.",
    )
    p.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Textual theme name (e.g. textual-dark).",
    )
    p.add_argument(
        "--import-words",
        type=str,
        action="append",
        default=None,
        help=(
            "Word list file (one per line). Optional tags via "
            "'term|tag1,tag2' or 'term<TAB>tag1,tag2'. Can repeat."
        ),
    )
    p.add_argument(
        "--remove-words",
        type=str,
        action="append",
        default=None,
        help=(
            "Word list file (one per line) to remove from the deck. "
            "If tags are present, they are ignored. Can repeat."
        ),
    )
    args = p.parse_args()

    default_db = str(Path.home() / ".yourei" / "yourei.sqlite3")
    default_config = Path.home() / ".yourei" / "config.toml"
    config_path = Path(args.config).expanduser() if args.config else default_config
    try:
        config = _load_toml_config(config_path, "yourei")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    db = args.db or _coerce_str(config.get("db")) or default_db
    source = args.source or _coerce_str(config.get("source")) or "English"
    target = args.target or _coerce_str(config.get("target")) or "French"
    model = args.model or _coerce_str(config.get("model")) or "llama3.1"
    ollama_url = (
        args.ollama_url
        or _coerce_str(config.get("ollama_url"))
        or "http://localhost:11434"
    )
    sentences = (
        args.sentences
        if args.sentences is not None
        else _coerce_int(config.get("sentences")) or 3
    )
    theme = args.theme or _coerce_str(config.get("theme"))
    import_words = args.import_words if args.import_words is not None else []
    remove_words = args.remove_words if args.remove_words is not None else []

    db_path = Path(db).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    ensure_db(conn)

    for words_path in remove_words:
        path = Path(words_path).expanduser()
        for term, _tags in iter_words_file(path):
            delete_card_by_term(
                conn, front=term, source_lang=source, target_lang=target
            )

    for words_path in import_words:
        path = Path(words_path).expanduser()
        for term, tags in iter_words_file(path):
            add_or_update_card(conn, front=term, source_lang=source, target_lang=target)
            if tags:
                card_id = get_card_id(
                    conn, front=term, source_lang=source, target_lang=target
                )
                if card_id is None:
                    continue
                for tag in tags:
                    add_tag_to_card(conn, card_id, tag)

    app = Yourei(
        conn=conn,
        source_lang=source,
        target_lang=target,
        model=model,
        ollama_url=ollama_url,
        sentences=sentences,
        theme=theme,
        config_path=config_path,
        db_path=db_path,
    )

    app.run()


if __name__ == "__main__":
    main()
