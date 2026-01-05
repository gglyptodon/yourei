#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import httpx
from textual.app import App, ComposeResult
from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.reactive import reactive

from textual.widgets import Button, Footer, Header, Input, Static, TextArea

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
    """
    Calls Ollama /api/generate and returns the full response text.
    """
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.4,
        },
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        try:
            r = await client.post(url, json=payload)
        except httpx.RequestError as e:
            raise OllamaError(f"Could not connect to Ollama at {base_url}: {e}") from e

    if r.status_code != 200:
        raise OllamaError(f"Ollama error {r.status_code}: {r.text}")

    data = r.json()
    # Ollama returns {"response": "...", ...}
    resp = data.get("response")
    if not isinstance(resp, str) or not resp.strip():
        raise OllamaError(f"Unexpected Ollama response: {data}")
    return resp.strip()


# ----------------------------
# Prompting
# ----------------------------


def build_system_prompt() -> str:
    return (
        "You are a language tutor. Be concise and helpful. "
        "Return ONLY valid JSON, no markdown, no code fences, no commentary."
    )


def build_user_prompt(
    word: str, source_lang: str, target_lang: str, n_sentences: int
) -> str:
    # Ask for a strict JSON schema.
    return f"""
Create a small learning card for the word.

Constraints:
- Source language: {source_lang}
- Target language: {target_lang}
- Word: "{word}"
- Provide a natural, context-appropriate translation (not a dictionary list).
- Provide {n_sentences} short example sentences in {source_lang} that use the word naturally.
- For each example sentence, provide its translation in {target_lang}.
- Keep sentences short and realistic (everyday language).

Return JSON with this exact schema:
{{
  "word": string,
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
    """
    Try to parse JSON even if the model adds leading/trailing text.
    We keep it simple: find first '{' and last '}' and parse.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return JSON.")
    chunk = raw[start : end + 1]
    return json.loads(chunk)


# ----------------------------
# Data
# ----------------------------


@dataclass
class Card:
    word: str
    translation: str
    examples: List[dict]  # {"source":..., "target":...}
    debug_data: dict


def _parse_term_line(line: str) -> Optional[str]:
    t = line.strip()
    if not t or t.startswith("#"):
        return None
    if "|" in t:
        term_raw, _tags_raw = t.split("|", 1)
    elif "\t" in t:
        term_raw, _tags_raw = t.split("\t", 1)
    else:
        term_raw = t
    term = term_raw.strip()
    return term if term else None


def load_words(path: Path) -> List[str]:
    words: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = _parse_term_line(line)
        if term:
            words.append(term)
    if not words:
        raise ValueError(f"No words found in {path}")
    return words


def format_card_for_right_panel(card: Card, target_lang: str, source_lang: str) -> str:
    lines = []
    lines.append(f"[{source_lang} → {target_lang}]")
    lines.append(f"Translation: {card.translation}")
    lines.append("Examples:")
    for i, ex in enumerate(card.examples, 1):
        s = (ex.get("source") or "").strip()
        t = (ex.get("target") or "").strip()
        lines.append(f"{i}. {s}")
        lines.append(f"   → {t}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ----------------------------
# Textual UI
# ----------------------------


class CardReady(Message):
    def __init__(self, card: Card) -> None:
        super().__init__()
        self.card = card


class ErrorMsg(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class LeftPanel(Static):
    word: reactive[str] = reactive("")
    info: reactive[str] = reactive("")

    def render(self) -> str:
        return f"{self.word}\n\n{self.info}".strip()


class AddWord(Message):
    def __init__(self, word: str) -> None:
        super().__init__()
        self.word = word


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

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="dialog"):
            yield Static("Add a new word/phrase:")
            yield Input(
                placeholder="e.g. nevertheless / to figure out / prendre en compte",
                id="word_input",
            )
            with Horizontal(id="buttons"):
                yield Button("Add", id="add", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#word_input", Input).focus()

    def _finish(self, word: str | None) -> None:
        if word:
            self.app.post_message(AddWord(word))
        self.dismiss(None)

    @on(Button.Pressed)
    def _on_button(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self._finish(None)
            return
        if event.button.id == "add":
            word = self.query_one("#word_input", Input).value.strip()
            self._finish(word if word else None)

    @on(Input.Submitted)
    def _on_submit(self, event: Input.Submitted) -> None:
        word = event.value.strip()
        self._finish(word if word else None)


class LangTui(App):
    CSS = """
Screen {
    layout: vertical;
}
#main {
    height: 1fr;
}
#left {
    width: 35%;
    border: round $secondary;
    padding: 1 2;
}
#right {
    width: 65%;
    border: round $secondary;
    padding: 1 2;
    height: 1fr;
}
.dim {
    color: $text-muted;
}
"""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("n", "next_word", "Next"),
        Binding("b", "prev_word", "Prev"),
        Binding("g", "regen", "Regenerate"),
        Binding("a", "add_word", "Add…"),
    ]

    def __init__(
        self,
        *,
        words: List[str],
        source_lang: str,
        target_lang: str,
        model: str,
        ollama_url: str,
        sentences: int,
    ) -> None:
        super().__init__()
        self.words: List[str] = words
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model = model
        self.ollama_url = ollama_url
        self.sentences = sentences

        self.index = 0
        self.cache: dict[str, Card] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            yield LeftPanel(id="left")
            yield TextArea("", id="right", read_only=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Yourei-Lite"
        self.call_later(self._show_current)

    def _current_word(self) -> str:
        if not self.words:
            return ""
        return self.words[self.index]

    def _advance_to_next_unknown(self, direction: int) -> None:
        if not self.words:
            return
        n = len(self.words)

        self.index = (self.index + direction) % n

    def _update_left_panel(self, status: str) -> None:
        left = self.query_one("#left", LeftPanel)
        word = self._current_word() if self.words else "(no words)"
        left.word = word
        left.info = (
            f"Model: {self.model}\n"
            f"Ollama: {self.ollama_url}\n"
            f"Words: {len(self.words)}\n\n"
            f"{status}\n\n"
            f"[dim]• a add… • n next • b prev • g regenerate • q quit[/dim]"
        )

    def _set_right_panel_text(self, text: str) -> None:
        right = self.query_one("#right", TextArea)
        right.load_text(text)

    def _show_current(self) -> None:
        if not self.words:
            self._update_left_panel("No words loaded.")
            self._set_right_panel_text("Press 'a' to add a new word.\n")
            return
        word = self._current_word()
        if word in self.cache:
            card = self.cache[word]
            self._update_left_panel("Loaded from cache.")
            self._set_right_panel_text(
                format_card_for_right_panel(card, self.target_lang, self.source_lang)
            )
        else:
            self._update_left_panel("Generating…")
            self._set_right_panel_text(
                "Generating translation + examples from local LLM…"
            )
            asyncio.create_task(self._generate_for_word(word))

    async def _generate_for_word(self, word: str) -> None:
        try:
            system = build_system_prompt()
            prompt = build_user_prompt(
                word, self.source_lang, self.target_lang, self.sentences
            )
            raw = await ollama_generate(
                base_url=self.ollama_url,
                model=self.model,
                prompt=prompt,
                system=system,
            )
            data = parse_card_json(raw)

            card = Card(
                word=str(data.get("word", word)),
                translation=str(data.get("translation", "")).strip(),
                examples=list(data.get("examples", []))
                if isinstance(data.get("examples", []), list)
                else [],
                debug_data=data,
            )
            if not card.translation:
                raise ValueError("Missing translation in JSON.")
            if not card.examples:
                raise ValueError("Missing examples in JSON.")

            self.cache[word] = card
            self.post_message(CardReady(card))
        except Exception as e:
            self.post_message(ErrorMsg(str(e)))

    def on_card_ready(self, msg: CardReady) -> None:
        # Only display if still on that word
        if msg.card.word != self._current_word():
            # still cache is useful
            pass
        self._update_left_panel("Generated.")
        self._set_right_panel_text(
            format_card_for_right_panel(msg.card, self.target_lang, self.source_lang)
        )

    def on_error_msg(self, msg: ErrorMsg) -> None:
        self._update_left_panel("Error.")
        self._set_right_panel_text(
            f"Error:\n{msg.text}\n\nTip: ensure `ollama serve` is running and the model name is correct."
        )

    # Actions
    def action_next_word(self) -> None:
        if not self.words:
            return
        self._advance_to_next_unknown(+1)
        self._show_current()

    def action_prev_word(self) -> None:
        if not self.words:
            return
        self._advance_to_next_unknown(-1)
        self._show_current()

    def action_regen(self) -> None:
        if not self.words:
            return
        word = self._current_word()
        self.cache.pop(word, None)
        self._show_current()

    def action_add_word(self) -> None:
        self.push_screen(AddWordModal())

    def on_add_word(self, msg: AddWord) -> None:
        word = msg.word.strip()
        if not word:
            return
        self.words.append(word)
        self.index = len(self.words) - 1
        self._show_current()


# ----------------------------
# CLI
# ----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local TUI vocabulary example generator using Ollama."
    )
    parser.add_argument(
        "--words",
        type=str,
        default=None,
        help="Path to a text file with one word per line.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="English",
        help="Source language label (e.g. English).",
    )
    parser.add_argument(
        "--target", type=str, default="French", help="Target language (e.g. French)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1",
        help="Ollama model name (e.g. llama3.1, mistral, qwen2.5).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=3,
        help="Number of example sentences to generate.",
    )
    args = parser.parse_args()

    words: List[str] = []
    if args.words:
        words_path = Path(args.words).expanduser()
        words = load_words(words_path)

    app = LangTui(
        words=words,
        source_lang=args.source,
        target_lang=args.target,
        model=args.model,
        ollama_url=args.ollama_url,
        sentences=args.sentences,
    )
    app.run()


if __name__ == "__main__":
    main()
