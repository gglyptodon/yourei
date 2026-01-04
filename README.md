Yourei
======

Terminal UIs for learning vocabulary with a local LLM (Ollama). This repo
includes:

- `yourei.py`: spaced repetition system with decks, tags,
  search, and import/remove helpers.
- `yourei-lite.py`: yourei-lite, a lightweight word list trainer that generates translations
  and examples on demand.

Requirements
------------

- Python 3.10+
- Ollama running locally (`ollama serve`)
- Python deps: `textual`, `httpx`

Install dependencies (example with uv):

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Yourei
------

Start the app:

```bash
yourei
```

Common options:

```bash
yourei --db ~/.yourei/yourei.sqlite3 --source English --target French \
  --model llama3.1 --ollama-url http://localhost:11434 --sentences 3 --theme textual-dark
```

TOML config (defaults to `~/.yourei/config.toml`, CLI overrides config):

```toml
# ~/.yourei/config.toml
# Either top-level keys or a [yourei] section are supported.
[yourei]
db = "~/.yourei/yourei.sqlite3"
source = "English"
target = "French"
model = "llama3.1"
ollama_url = "http://localhost:11434"
sentences = 3
theme = "textual-dark"
```

Note: TOML config parsing needs Python 3.11+ or `tomli` installed for Python 3.10.
Import/remove word lists are CLI-only (not stored in config).
Press `p` in the TUI to save current settings to the TOML file.

Import words (with optional tags):

```bash
yourei --import-words words.txt
```

The import format supports tags per line:

```
take a photo | verbs, travel
do the dishes | verbs, day-to-day
```

Remove words in bulk:

```bash
yourei --remove-words remove.txt
```

Key bindings (Yourei):

- `0-5` grade current card
- `g` (re)generate from LLM
- `u` populate translations
- `n` skip
- `b` back
- `x` suspend
- `a` add word
- `A` add word (reverse)
- `d` delete
- `t` filter by tag
- `l` add tag
- `r` remove tag
- `s` search deck
- `c` switch deck
- `m` switch model
- `v` toggle review mode
- `space` reveal translation (in review mode)
- `p` save config to TOML
- `y` toggle due-today-only
- `q` quit
- `Esc` close dialogs

<img width="1373" height="758" alt="yourei_example" src="https://github.com/user-attachments/assets/8bd71232-f0cf-437e-9779-5aff89df5a88" />


Yourei-lite
-----------

Start the app (optionally provide a word list):

```bash
yourei-lite
yourei-lite --words words.txt
```

Common options:

```bash
yourei-lite --source English --target French \
  --model llama3.1 --ollama-url http://localhost:11434 --sentences 3
```

Add words directly in the TUI with the `a` hotkey. Words added in the TUI are not persisted.

Key bindings (Yourei-lite):

- `q` quit
- `n` next word
- `b` previous word
- `g` regenerate
- `a` add word
