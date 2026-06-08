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

Python 3.11 is currently used in this workspace.

Dependencies:

- `fastapi`
- `uvicorn`
- `pygame`

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
  "seed": 7
}
```

Fields:

- `width`: board width
- `height`: board height
- `mine_density`: optional, must be between `0` and `1`
- `seed`: optional, useful for deterministic test runs

### Get Current Game State

`GET /games/{game_id}`

Returns:

- game metadata
- score
- move count
- visible board state
- status

### Submit A Move

`POST /games/{game_id}/moves`

Request body:

```json
{
  "action": "reveal",
  "x": 0,
  "y": 0
}
```

Allowed actions:

- `reveal`
- `flag`

The response includes:

- updated game state
- score delta for the last move
- move metadata

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
