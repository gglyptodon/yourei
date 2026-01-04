#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

DEFAULT_DB = Path.home() / ".yourei" / "yourei.sqlite3"
DEFAULT_SOURCE = "English"
DEFAULT_TARGET = "French"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export generated translations (with tags) as importable source terms "
            "for yourei.py --import-words."
        )
    )
    p.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB),
        help="SQLite DB path (default: ~/.yourei/yourei.sqlite3).",
    )
    p.add_argument(
        "--source", type=str, default=DEFAULT_SOURCE, help="Source language label."
    )
    p.add_argument(
        "--target", type=str, default=DEFAULT_TARGET, help="Target language label."
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output word list path.",
    )
    return p.parse_args()


def fetch_translations(
    conn: sqlite3.Connection, *, source_lang: str, target_lang: str
) -> List[Tuple[str, List[str]]]:
    rows = conn.execute(
        """
        SELECT cards.id, cards.llm_translation, tags.name
        FROM cards
        LEFT JOIN card_tags ON card_tags.card_id = cards.id
        LEFT JOIN tags ON tags.id = card_tags.tag_id
        WHERE cards.source_lang=? AND cards.target_lang=?
          AND cards.llm_translation IS NOT NULL
          AND TRIM(cards.llm_translation) != ''
        ORDER BY cards.id ASC, tags.name COLLATE NOCASE
        """,
        (source_lang, target_lang),
    ).fetchall()

    grouped: List[Tuple[str, List[str]]] = []
    current_id: Optional[int] = None
    current_translation = ""
    current_tags: List[str] = []

    for card_id, translation, tag in rows:
        if current_id != card_id:
            if current_id is not None:
                grouped.append((current_translation, current_tags))
            current_id = int(card_id)
            current_translation = str(translation).strip()
            current_tags = []
        if tag is not None:
            current_tags.append(str(tag))

    if current_id is not None:
        grouped.append((current_translation, current_tags))

    return grouped


def format_line(term: str, tags: List[str]) -> str:
    if tags:
        return f"{term}|{','.join(tags)}"
    return term


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        rows = fetch_translations(
            conn, source_lang=args.source, target_lang=args.target
        )
    finally:
        conn.close()

    lines = [format_line(term, tags) for term, tags in rows if term]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
