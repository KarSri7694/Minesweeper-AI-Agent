# Minesweeper AI Agent

A reproducible Minesweeper environment with a FastAPI service, terminal and
pygame interfaces, and a hybrid playing policy: deterministic constraint logic
first, exact local mine probabilities second, and an optional schema-constrained
Ollama LLM when the board remains ambiguous.

## What is here

- A single game engine with seeded games, safe first reveal, scoring, compact
  and detailed state representations, and restorable snapshots.
- Terminal, GUI, HTTP API, and GUI spectator modes using the same engine.
- A hybrid agent that prevents malformed LLM actions and repeated actions on an
  unchanged board.
- Logic-labelled JSONL data generation for SFT/offline experiments.
- A fixed-seed benchmark for measuring policy changes before and after training.

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Terminal game
APP_MODE=terminal python src/game.py --width 9 --height 9 --mine-density 0.15

# API server
APP_MODE=api python src/game.py --host 127.0.0.1 --port 8000
```

For the full API and GUI guide, see [documentation.md](documentation.md).

## Agent and evaluation

The standard hybrid policy uses visible-board constraints, then probabilities,
then a safe fallback. It uses the LLM only for ambiguous states or ties unless
`LLM_FROM_START=1` is set. Configure Ollama with `OLLAMA_HOST`,
`OLLAMA_MODEL`, and optionally `OLLAMA_TIMEOUT_SECONDS`.

```bash
# Reproducible logic-labelled examples
PYTHONPATH=src python src/dataset_generator.py --samples 1000 \
  --output datasets/logic_moves.jsonl

# Fixed-seed baseline benchmark
python scripts/benchmark.py --games 100 --width 6 --height 6 \
  --mine-density 0.15 --seed 10000 --output datasets/benchmark.json

# Test the core engine, API, and agent helpers
python3 -m unittest discover -s tests -v
```

Track at least win rate, average score, flag precision, invalid action rate,
repeated-action rate, and latency for each model or prompt change. Always keep
the seed range used for evaluation separate from training data.

## Project layout

```text
src/minesweeper/engine.py  Shared game rules and snapshot support
src/minesweeper/api.py     FastAPI game service
src/llm_parser.py          Hybrid policy and Ollama integration
src/dataset_generator.py   Logic-labelled JSONL generator
scripts/benchmark.py       Fixed-seed policy benchmark
tests/                      Engine, API, and agent unit tests
```
