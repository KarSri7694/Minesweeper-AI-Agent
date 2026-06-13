# Minesweeper LLM Documentation

## Overview

This project provides a Minesweeper game with three access modes built on the same game engine:

- Terminal mode for manual text-based play
- API mode for AI agents or other programmatic clients
- GUI mode using `pygame`

The game supports variable board sizes such as `5x5`, `9x9`, `12x12`, `15x15`, and other rectangular sizes within the configured validation limits.

## Core Rules

- First reveal is always safe
- Supported actions are `reveal` and `flag`
- Revealing a mine ends the game immediately
- A win requires all safe tiles to be revealed and all mines to be correctly flagged

## Scoring

- Reveal a safe tile: `+1` per revealed safe tile
- Flag a mine correctly: `+2`
- Flag a non-mine tile: `-2`
- Reveal a mine: game ends
- Win the full game: `+50` bonus

Notes:

- Removing a flag does not add or subtract score
- Re-flagging the same tile does not repeatedly award or deduct points

## Project Structure

- `src/game.py`
  Main entrypoint
- `src/minesweeper/engine.py`
  Shared Minesweeper engine, scoring, state, and session manager
- `src/minesweeper/api.py`
  FastAPI application and HTTP endpoints
- `src/minesweeper/cli.py`
  Terminal renderer and input loop
- `src/minesweeper/gui.py`
  `pygame` graphical interface
- `src/minesweeper/app.py`
  Mode switch for terminal, API, and GUI
- `tests/`
  Unit tests for engine and API behavior

## Requirements

Python 3.13 is currently used in this workspace.

Dependencies:

- `fastapi`
- `numpy`
- `openai`
- `uvicorn`
- `pygame`
- `pytest`
- `requests`

## Running The App

### PowerShell

Set the mode with `$env:APP_MODE`.

### Terminal Mode

```powershell
$env:APP_MODE = "terminal"
venv\Scripts\python src\game.py --width 9 --height 9 --mine-density 0.15
```

Commands inside terminal mode:

- `reveal x y`
- `flag x y`
- `quit`

Example:

```text
reveal 3 4
flag 5 1
```

### GUI Mode

```powershell
$env:APP_MODE = "gui"
venv\Scripts\python src\game.py --width 9 --height 9 --mine-density 0.15
```

GUI controls:

- Left click: reveal tile
- Right click: flag or unflag tile
- `R`: restart the current board
- `Esc`: quit

### GUI Spectator Mode For API Games

If the AI is playing through the HTTP API, you can open the `pygame` window as a read-only spectator for that same `game_id`.

Example:

```powershell
$env:APP_MODE = "api"
venv\Scripts\python src\game.py --host 127.0.0.1 --port 8000
```

In another PowerShell window, start the AI client and note the `game_id`, then open the spectator:

```powershell
$env:APP_MODE = "gui"
venv\Scripts\python src\game.py --spectate-game-id YOUR_GAME_ID --api-base-url http://127.0.0.1:8000
```

Notes:

- Spectator mode refreshes the board automatically from the API
- Mouse play is disabled while spectating
- `R` restart is disabled while spectating
- `Esc` closes the spectator window

### API Mode

```powershell
$env:APP_MODE = "api"
venv\Scripts\python src\game.py --host 127.0.0.1 --port 8000
```

API base URL:

```text
http://127.0.0.1:8000
```

## API Endpoints

### Health Check

`GET /health`

Response:

```json
{
  "status": "ok"
}
```

### Create Game

`POST /games`

Request body:

```json
{
  "width": 9,
  "height": 9,
  "mine_density": 0.15,
  "seed": 7,
  "output_format": "detailed"
}
```

Fields:

- `width`: board width
- `height`: board height
- `mine_density`: optional, must be between `0` and `1`
- `seed`: optional, useful for deterministic test runs
- `output_format`: optional, either `detailed` or `compact`

### Get Current Game State

`GET /games/{game_id}`

Optional query parameter:

- `output_format=detailed|compact`

Returns:

- game metadata
- score
- move count
- visible board state
- status

### Get Current Game State With Request Body

`POST /games/{game_id}/state`

Request body:

```json
{
  "output_format": "compact"
}
```

### Submit A Move

`POST /games/{game_id}/moves`

Request body:

```json
{
  "action": "reveal",
  "x": 0,
  "y": 0,
  "output_format": "compact"
}
```

Allowed actions:

- `reveal`
- `flag`

The response includes:

- updated game state
- score delta for the last move
- move metadata
- requested board output format

## Game State Shape

The serialized game state includes fields such as:

- `game_id`
- `status`
- `width`
- `height`
- `mine_count`
- `flagged_count`
- `score`
- `move_count`
- `end_reason`
- `board`

During active play, hidden mines are not exposed through the API.

### Output Formats

#### `detailed`

This is the original response shape where `board` is a 2D array of tile objects.

#### `compact`

In compact mode, `board` is a 2D array of strings:

- `.` for unrevealed tiles
- `F` for flagged tiles
- `_` for revealed tiles with zero adjacent mines
- `"1"` to `"8"` for revealed numbered tiles
- `B` for bombs visible after a loss

## Dataset And Fine-Tuning Workflow

Run commands from the repository root on macOS using `venv/bin/python3`.

### Generate Raw Transition Data

```bash
venv/bin/python3 src/minesweeper/solve_algo.py --rows 9 --cols 9 --mines 0.15 --games 1000 --filename games_9x9_1000.jsonl --output datasets/generated
```

This writes transition records to `datasets/generated/games_9x9_1000.jsonl`.

### Convert To Chat Fine-Tuning JSONL

Create the full dataset:

```bash
venv/bin/python3 src/dataset_generator.py --input datasets/generated/games_9x9_1000.jsonl --output datasets/finetune/minesweeper_9x9_1000_all_finetune.jsonl
```

Create a safer dataset that skips moves which immediately led to a loss:

```bash
venv/bin/python3 src/dataset_generator.py --input datasets/generated/games_9x9_1000.jsonl --output datasets/finetune/minesweeper_9x9_1000_safe_finetune.jsonl --skip-terminal-losses
```

Inspect examples:

```bash
venv/bin/python3 scripts/inspect_finetune_data.py --input datasets/finetune/minesweeper_9x9_1000_safe_finetune.jsonl --count 3
```

Check action and coordinate distribution:

```bash
venv/bin/python3 scripts/check_action_distribution.py --input datasets/finetune/minesweeper_9x9_1000_safe_finetune.jsonl
```

### Split Train And Validation Data

```bash
venv/bin/python3 scripts/split_finetune_data.py --input datasets/finetune/minesweeper_9x9_1000_safe_finetune.jsonl --train-output datasets/finetune/train.jsonl --val-output datasets/finetune/val.jsonl --val-ratio 0.1 --seed 42
```

### Start An OpenAI Fine-Tuning Job

Do not put API keys in files. Export `OPENAI_API_KEY` in your shell or prefix the command locally.

```bash
OPENAI_API_KEY=... venv/bin/python3 scripts/start_openai_finetune.py --train datasets/finetune/train.jsonl --val datasets/finetune/val.jsonl --model <base-model-name> --suffix minesweeper-9x9
```

The script uploads the train and validation JSONL files with `purpose="fine-tune"`, starts a supervised fine-tuning job, prints the uploaded file IDs and job ID, and writes metadata to:

```text
datasets/finetune/openai_finetune_job.json
```

### Check Fine-Tuning Status

Read the latest saved job metadata:

```bash
OPENAI_API_KEY=... venv/bin/python3 scripts/check_openai_finetune.py
```

Or pass a job ID directly:

```bash
OPENAI_API_KEY=... venv/bin/python3 scripts/check_openai_finetune.py --job-id <fine-tuning-job-id>
```

### Evaluate A Fine-Tuned Model

Keep evaluation small at first to control API cost:

```bash
OPENAI_API_KEY=... venv/bin/python3 scripts/evaluate_finetuned_model.py --model <fine-tuned-model-name> --input datasets/finetune/val.jsonl --max-examples 50
```

The evaluator reports exact match accuracy, action accuracy, coordinate accuracy, and invalid JSON count.

### Test The Tooling

```bash
venv/bin/python3 -m pytest
```

Example compact board:

```text
. . . 1
_ F 1 .
_ 2 3 _
_ _ _ _
```

## Testing

Run the test suite:

```powershell
venv\Scripts\python -m unittest discover -s tests -v
```

Optional compile check:

```powershell
venv\Scripts\python -m compileall src tests
```

## Common Issues

### GUI Does Not Open

If you are using PowerShell, do not use `set APP_MODE=gui`.

Use:

```powershell
$env:APP_MODE = "gui"
venv\Scripts\python src\game.py
```

### Wrong App Mode

You can inspect the current mode in PowerShell:

```powershell
$env:APP_MODE
```

## Current Limitations

- Game sessions in API mode are stored in memory only
- Sessions are lost when the process exits
- GUI has no in-window settings panel yet
- There is no persistent leaderboard or replay storage yet
