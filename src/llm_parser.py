import requests
import json
import ast
from pathlib import Path
import asyncio
import os
import random
from dataclasses import dataclass, field

import pygame

from minesweeper.engine import GameConfig, GameEngine
from minesweeper.gui import (
    HUD_HEIGHT,
    PADDING,
    _choose_tile_size,
    _create_screen,
    _draw,
)

game_url = os.getenv("GAME_API_BASE_URL", "http://localhost:8000/")
game_backend = os.getenv("GAME_BACKEND", "direct").lower()
HIDDEN_TILE = "."
FLAGGED_TILE = "F"
MAX_PROBABILITY_COMPONENT_SIZE = 18
MODEL_GUI_ENABLED = os.getenv("MODEL_GUI", "0") == "1"
MODEL_GUI_STEP_MS = int(os.getenv("MODEL_GUI_STEP_MS", "250"))
DEBUG_RUN_ENABLED = os.getenv("DEBUG_RUN", "0") == "1"


@dataclass
class MoveGuard:
    """Reject duplicate moves for an unchanged visible board.

    This is deliberately keyed by the visible board rather than only by the
    coordinate: returning to a coordinate may be valid after the board has
    changed, but repeating an action without a state transition is a loop.
    """

    attempted_moves: set[tuple[str, tuple[tuple[str, ...], ...], str, int, int]] = field(
        default_factory=set
    )

    def accept(self, move: dict | None, game_state: dict) -> bool:
        if move is None:
            return False
        board = game_state.get("board") or []
        board_key = tuple(tuple(str(cell) for cell in row) for row in board)
        key = (
            str(game_state.get("game_id", "")),
            board_key,
            str(move.get("action")),
            int(move.get("x", -1)),
            int(move.get("y", -1)),
        )
        if key in self.attempted_moves:
            return False
        self.attempted_moves.add(key)
        return True


def get_ollama_base_url():
    return (
        os.getenv("OLLAMA_HOST")
        or os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_URL")
        or "http://127.0.0.1:11434"
    ).rstrip("/")


def get_ollama_model():
    return os.getenv("OLLAMA_MODEL", "minesweeper-grpo")


def get_ollama_timeout_seconds():
    return int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))


def get_ollama_num_predict():
    # A move normally needs far fewer tokens, but leave room for multi-digit
    # coordinates and the closing brace so generation cannot truncate JSON.
    return int(os.getenv("OLLAMA_NUM_PREDICT", "64"))


def use_llm_from_start():
    return os.getenv("LLM_FROM_START", "0") == "1"


def get_system_prompt():
    parent_dir = Path(__file__).parent.parent
    system_prompt = parent_dir / "system_prompt.md"
    with open(system_prompt, "r", encoding='utf-8') as f:
        return f.read()


def get_move_system_prompt():
    return (
        "You are playing Minesweeper. Return exactly one legal move as minified JSON only. "
        "Never explain. Never echo the board. "
        "Valid schema: {\"action\":\"reveal\"|\"flag\",\"x\":<int>,\"y\":<int>}."
    )


MOVE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "x", "y"],
    "properties": {
        "action": {"type": "string", "enum": ["reveal", "flag"]},
        "x": {"type": "integer"},
        "y": {"type": "integer"},
    },
}
    
def create_game():
    json_data = {
        "width": int(os.getenv("GAME_WIDTH", "12")),
        "height": int(os.getenv("GAME_HEIGHT", "12")),
        "mine_density": float(os.getenv("GAME_MINE_DENSITY", "0.15")),
        "seed": int(os.getenv("GAME_SEED", "7")),
        "output_format": "compact",
    }
    if game_backend == "direct":
        game = GameEngine(
            config=GameConfig(
                width=json_data["width"],
                height=json_data["height"],
                mine_density=json_data["mine_density"],
            ),
            rng=random.Random(json_data["seed"]),
        )
        print("Direct game created successfully!")
        print(
            "Board: "
            f"{json_data['width']}x{json_data['height']}, "
            f"density={json_data['mine_density']}, seed={json_data['seed']}"
        )
        print(f"Game ID: {game.game_id}")
        return game

    if game_backend != "api":
        raise ValueError("GAME_BACKEND must be either 'direct' or 'api'.")

    response = requests.post(game_url + "games/", json=json_data)
    if response.status_code == 200:
        print("Game created successfully!")
        print(
            "Board: "
            f"{json_data['width']}x{json_data['height']}, "
            f"density={json_data['mine_density']}, seed={json_data['seed']}"
        )
        game_id = response.json().get("game_id")
        print(f"Game ID: {game_id}")
        return game_id

def get_game_state(game_id):
    if isinstance(game_id, GameEngine):
        return game_id.compact_state()

    json_data = {
        "output_format": "compact"
    }
    response = requests.post(game_url + f"games/{game_id}/state", json=json_data)
    # print(response.json().get("board"))
    return response.json()
    
def make_move(game_id, action, x, y):
    if isinstance(game_id, GameEngine):
        try:
            if action == "reveal":
                game_id.reveal(x, y)
            elif action == "flag":
                game_id.flag(x, y)
            else:
                raise ValueError("Action must be either 'reveal' or 'flag'.")
        except ValueError as exc:
            print("Failed to make move.")
            print(exc)
            return None

        print("Move made successfully!")
        return game_id.compact_state()

    body = {
        "action": action,
        "x": x,
        "y": y,
        "output_format": "compact"
    }
    response = requests.post(game_url + f"games/{game_id}/moves/", json=body)
    if response.status_code == 200:
        print("Move made successfully!")
        return response.json()
    else:
        print("Failed to make move.")
        print(response.text)
        return None

def connect_to_llm():
    return {
        "base_url": get_ollama_base_url(),
        "model": get_ollama_model(),
        "timeout": get_ollama_timeout_seconds(),
    }

async def get_llm_response(llm, system_prompt, user_prompt):
    try:
        response = requests.post(
            f"{llm['base_url']}/api/generate",
            json={
                "model": llm["model"],
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": True,
                "think": False,
                # Ollama constrains decoding to this schema. Prompting alone
                # cannot reliably prevent prose, Markdown, or truncated JSON.
                "format": MOVE_JSON_SCHEMA,
                "options": {
                    "num_predict": get_ollama_num_predict(),
                },
            },
            timeout=llm["timeout"],
            stream=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"LLM request failed: {exc}")
        return None

    chunks = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        chunk = payload.get("response", "")
        if chunk:
            chunks.append(chunk)
            candidate = "".join(chunks).strip()
            try:
                parsed = _extract_json_object(candidate)
                if _is_move_payload(parsed):
                    return json.dumps(parsed, separators=(",", ":"))
            except json.JSONDecodeError:
                pass

        if payload.get("done"):
            break

    final_text = "".join(chunks).strip()
    try:
        parsed = _extract_json_object(final_text)
        if _is_move_payload(parsed):
            return json.dumps(parsed, separators=(",", ":"))
    except json.JSONDecodeError:
        pass
    return final_text

def parse_llm_response(response):
    response_json = _extract_json_object(response)
    normalized_move = _normalize_move_payload(response_json)
    return normalized_move.get("action"), normalized_move.get("row"), normalized_move.get("col")


def _extract_json_object(text):
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped.split("```json", 1)[1].split("```", 1)[0].strip()
    elif stripped.startswith("```"):
        stripped = stripped.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, dict):
            return parsed
    except (SyntaxError, ValueError):
        pass

    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
            return parsed
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(text[index:])
            if isinstance(parsed, dict):
                return parsed
        except (SyntaxError, ValueError):
            continue
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _is_move_payload(payload):
    normalized_move = _normalize_move_payload(payload)
    return (
        normalized_move is not None
        and normalized_move.get("action") in {"reveal", "flag"}
        and isinstance(normalized_move.get("row"), int)
        and isinstance(normalized_move.get("col"), int)
    )


def _normalize_move_payload(payload):
    if not isinstance(payload, dict):
        return None

    action = payload.get("action")
    row = payload.get("row")
    col = payload.get("col")

    if row is None and "y" in payload:
        row = payload.get("y")
    if col is None and "x" in payload:
        col = payload.get("x")

    try:
        normalized_move = {
            "action": str(action).lower().strip(),
            "row": int(row),
            "col": int(col),
        }
    except (TypeError, ValueError):
        return None

    return normalized_move


def neighbors_for(board, x, y):
    height = len(board)
    width = len(board[0]) if height else 0
    neighbors = []
    for ny in range(max(0, y - 1), min(height, y + 2)):
        for nx in range(max(0, x - 1), min(width, x + 2)):
            if nx == x and ny == y:
                continue
            neighbors.append((nx, ny))
    return neighbors


def is_revealed_number(value):
    return isinstance(value, str) and value.isdigit()


def get_constraints(board):
    constraints = []
    for y, row in enumerate(board):
        for x, value in enumerate(row):
            if not is_revealed_number(value):
                continue

            hidden_neighbors = set()
            flagged_neighbors = 0
            for nx, ny in neighbors_for(board, x, y):
                neighbor = board[ny][nx]
                if neighbor == HIDDEN_TILE:
                    hidden_neighbors.add((nx, ny))
                elif neighbor == FLAGGED_TILE:
                    flagged_neighbors += 1

            remaining_mines = int(value) - flagged_neighbors
            if hidden_neighbors and remaining_mines >= 0:
                constraints.append((hidden_neighbors, remaining_mines))
    return constraints


def deterministic_move(game_state):
    board = game_state.get("board") or []
    height = len(board)
    width = len(board[0]) if height else 0
    if not width or not height:
        return None

    if game_state.get("move_count", 0) == 0:
        return {"action": "reveal", "x": width // 2, "y": height // 2}

    constraints = get_constraints(board)

    for cells, remaining_mines in constraints:
        if remaining_mines == len(cells):
            x, y = sorted(cells)[0]
            return {"action": "flag", "x": x, "y": y}

    for cells, remaining_mines in constraints:
        if remaining_mines == 0:
            x, y = sorted(cells)[0]
            return {"action": "reveal", "x": x, "y": y}

    for left_cells, left_mines in constraints:
        for right_cells, right_mines in constraints:
            if left_cells == right_cells or not left_cells < right_cells:
                continue

            difference = right_cells - left_cells
            difference_mines = right_mines - left_mines
            if difference_mines == len(difference):
                x, y = sorted(difference)[0]
                return {"action": "flag", "x": x, "y": y}
            if difference_mines == 0:
                x, y = sorted(difference)[0]
                return {"action": "reveal", "x": x, "y": y}

    return None


def hidden_tiles_for(board):
    return [
        (x, y)
        for y, row in enumerate(board)
        for x, value in enumerate(row)
        if value == HIDDEN_TILE
    ]


def split_constraint_components(constraints):
    remaining_constraints = [
        (set(cells), remaining_mines)
        for cells, remaining_mines in constraints
        if cells
    ]
    components = []

    while remaining_constraints:
        component_constraints = [remaining_constraints.pop()]
        component_cells = set(component_constraints[0][0])
        changed = True
        while changed:
            changed = False
            next_remaining = []
            for cells, remaining_mines in remaining_constraints:
                if cells & component_cells:
                    component_constraints.append((cells, remaining_mines))
                    component_cells.update(cells)
                    changed = True
                else:
                    next_remaining.append((cells, remaining_mines))
            remaining_constraints = next_remaining
        components.append((component_cells, component_constraints))

    return components


def enumerate_component_probabilities(component_cells, component_constraints):
    ordered_cells = sorted(component_cells)
    if len(ordered_cells) > MAX_PROBABILITY_COMPONENT_SIZE:
        return None

    index_by_cell = {cell: index for index, cell in enumerate(ordered_cells)}
    indexed_constraints = [
        ([index_by_cell[cell] for cell in cells], remaining_mines)
        for cells, remaining_mines in component_constraints
    ]
    constraint_indexes_by_cell = [[] for _ in ordered_cells]
    for constraint_index, (indexes, _) in enumerate(indexed_constraints):
        for cell_index in indexes:
            constraint_indexes_by_cell[cell_index].append(constraint_index)

    assigned_mines = [0] * len(indexed_constraints)
    unassigned = [len(indexes) for indexes, _ in indexed_constraints]
    assignments = [0] * len(ordered_cells)
    valid_assignments = 0
    mine_counts = {cell: 0 for cell in ordered_cells}

    def can_assign(cell_index, value):
        for constraint_index in constraint_indexes_by_cell[cell_index]:
            _, required_mines = indexed_constraints[constraint_index]
            next_mines = assigned_mines[constraint_index] + value
            next_unassigned = unassigned[constraint_index] - 1
            if next_mines > required_mines:
                return False
            if next_mines + next_unassigned < required_mines:
                return False
        return True

    def apply_assignment(cell_index, value):
        for constraint_index in constraint_indexes_by_cell[cell_index]:
            assigned_mines[constraint_index] += value
            unassigned[constraint_index] -= 1

    def undo_assignment(cell_index, value):
        for constraint_index in constraint_indexes_by_cell[cell_index]:
            assigned_mines[constraint_index] -= value
            unassigned[constraint_index] += 1

    def backtrack(cell_index):
        nonlocal valid_assignments
        if cell_index == len(ordered_cells):
            if all(assigned_mines[index] == required for index, (_, required) in enumerate(indexed_constraints)):
                valid_assignments += 1
                for index, value in enumerate(assignments):
                    if value:
                        mine_counts[ordered_cells[index]] += 1
            return

        for value in (0, 1):
            if not can_assign(cell_index, value):
                continue
            assignments[cell_index] = value
            apply_assignment(cell_index, value)
            backtrack(cell_index + 1)
            undo_assignment(cell_index, value)

    backtrack(0)
    if valid_assignments == 0:
        return None

    return {
        cell: mine_count / valid_assignments
        for cell, mine_count in mine_counts.items()
    }


def probability_move(game_state):
    ranked = probability_candidates(game_state)
    if not ranked:
        return None
    x, y = ranked[0][0]
    print(f"Probability estimate for ({x}, {y}): {ranked[0][1]:.3f}")
    if DEBUG_RUN_ENABLED:
        pretty_candidates = ", ".join(
            f"({cx},{cy})={prob:.3f}" for (cx, cy), prob in ranked[:5]
        )
        print(f"[debug] top probability candidates: {pretty_candidates}")
    return {"action": "reveal", "x": x, "y": y}


def probability_candidates(game_state):
    board = game_state.get("board") or []
    hidden_tiles = hidden_tiles_for(board)
    if not hidden_tiles:
        return []

    constraints = get_constraints(board)
    probabilities = {}
    for component_cells, component_constraints in split_constraint_components(constraints):
        component_probabilities = enumerate_component_probabilities(
            component_cells,
            component_constraints,
        )
        if component_probabilities:
            probabilities.update(component_probabilities)

    if not probabilities:
        return []

    unconstrained_tiles = [cell for cell in hidden_tiles if cell not in probabilities]
    remaining_mines = game_state.get("mine_count", 0) - game_state.get("flagged_count", 0)
    if unconstrained_tiles and remaining_mines > 0:
        expected_frontier_mines = sum(probabilities.values())
        unconstrained_probability = max(
            0.0,
            min(1.0, (remaining_mines - expected_frontier_mines) / len(unconstrained_tiles)),
        )
        for cell in unconstrained_tiles:
            probabilities[cell] = unconstrained_probability

    return sorted(
        probabilities.items(),
        key=lambda item: (item[1], item[0][1], item[0][0]),
    )


def fallback_guess(game_state):
    board = game_state.get("board") or []
    hidden_tiles = hidden_tiles_for(board)
    if not hidden_tiles:
        return None

    x, y = min(hidden_tiles, key=lambda cell: (cell[1], cell[0]))
    return {"action": "reveal", "x": x, "y": y}


def build_user_prompt(game_state):
    visible_state = {
        "width": game_state.get("width"),
        "height": game_state.get("height"),
        "mine_count": game_state.get("mine_count"),
        "flagged_count": game_state.get("flagged_count"),
        "score": game_state.get("score"),
        "move_count": game_state.get("move_count"),
        "board": game_state.get("board") or [],
        "output_format": game_state.get("output_format", "compact"),
    }
    return (
        "Return exactly one move as raw minified JSON with keys action, x, y. "
        "Do not explain, do not write code, and do not include markdown.\n"
        f"{json.dumps(visible_state, separators=(',', ':'))}"
    )


def build_probability_tiebreak_prompt(game_state, candidates):
    visible_state = {
        "board": game_state.get("board"),
        "candidate_reveals": [
            {"x": x, "y": y, "estimated_mine_probability": probability}
            for (x, y), probability in candidates
        ],
    }
    return (
        "Choose exactly one reveal from candidate_reveals. "
        "Return only minified JSON using action, x, and y.\n"
        f"{json.dumps(visible_state)}"
    )


def build_retry_prompt(game_state, invalid_response):
    visible_state = {
        "width": game_state.get("width"),
        "height": game_state.get("height"),
        "mine_count": game_state.get("mine_count"),
        "flagged_count": game_state.get("flagged_count"),
        "score": game_state.get("score"),
        "move_count": game_state.get("move_count"),
        "board": game_state.get("board") or [],
        "output_format": game_state.get("output_format", "compact"),
    }
    return (
        "Your previous reply was invalid. Reply again using only minified JSON matching "
        "{\"action\":\"reveal\"|\"flag\",\"x\":<int>,\"y\":<int>}.\n"
        f"Invalid reply: {invalid_response}\n"
        f"Board state: {json.dumps(visible_state, separators=(',', ':'))}"
    )


def validate_move(move, game_state):
    normalized_move = _normalize_move_payload(move)
    if normalized_move is None:
        return None

    action = normalized_move.get("action")
    row = normalized_move.get("row")
    col = normalized_move.get("col")
    board = game_state.get("board") or []
    height = len(board)
    width = len(board[0]) if height else 0

    if action not in {"reveal", "flag"}:
        return None
    if col < 0 or row < 0 or col >= width or row >= height:
        return None
    if board[row][col] != HIDDEN_TILE:
        return None
    return {"action": action, "x": col, "y": row, "row": row, "col": col}


def debug_board_text(game_state):
    board = game_state.get("board") or []
    return "\n".join(" ".join(row) for row in board)


def debug_state_summary(game_state):
    board = game_state.get("board") or []
    hidden_count = sum(cell == HIDDEN_TILE for row in board for cell in row)
    flagged_count = sum(cell == FLAGGED_TILE for row in board for cell in row)
    return (
        f"status={game_state.get('status')} "
        f"score={game_state.get('score')} "
        f"moves={game_state.get('move_count')} "
        f"hidden={hidden_count} flagged={flagged_count}/{game_state.get('mine_count')}"
    )


def debug_log_state(label, game_state):
    if not DEBUG_RUN_ENABLED:
        return
    print(f"[debug] {label}: {debug_state_summary(game_state)}")
    print(debug_board_text(game_state))


async def choose_move_with_llm(llm, game_state):
    prompts = [
        build_user_prompt(game_state),
    ]
    assistant_text = ""

    for attempt_index, prompt in enumerate(prompts):
        response = await get_llm_response(llm, get_system_prompt(), prompt)
        assistant_text = response or ""
        if assistant_text:
            print(assistant_text)

        try:
            action, row, col = parse_llm_response(assistant_text)
            move = validate_move({"action": action, "row": row, "col": col}, game_state)
            if move is not None:
                return move
        except json.decoder.JSONDecodeError:
            pass

        if attempt_index == 0:
            prompts.append(build_retry_prompt(game_state, assistant_text))

    return None


def init_model_gui(game):
    if not MODEL_GUI_ENABLED or not isinstance(game, GameEngine):
        return None

    pygame.init()
    pygame.display.set_caption("Minesweeper LLM Direct")
    width = game.config.width
    height = game.config.height
    tile_size = _choose_tile_size(width, height)
    return {
        "screen": _create_screen(width, height),
        "clock": pygame.time.Clock(),
        "title_font": pygame.font.SysFont("consolas", 28, bold=True),
        "body_font": pygame.font.SysFont("consolas", 20),
        "tile_font": pygame.font.SysFont("consolas", max(18, tile_size // 2), bold=True),
        "tile_size": tile_size,
        "running": True,
    }


def render_model_gui(gui, game, message):
    if gui is None or not gui["running"]:
        return True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            gui["running"] = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            gui["running"] = False

    if not gui["running"]:
        return False

    _draw(
        screen=gui["screen"],
        state=game.visible_state(),
        tile_size=gui["tile_size"],
        board_origin_x=PADDING,
        board_origin_y=HUD_HEIGHT,
        title_font=gui["title_font"],
        body_font=gui["body_font"],
        tile_font=gui["tile_font"],
        message=message,
        spectator_mode=False,
    )
    pygame.display.flip()
    pygame.time.wait(MODEL_GUI_STEP_MS)
    gui["clock"].tick(60)
    return True


async def main():
    game_id = create_game()
    game_state = get_game_state(game_id)
    debug_log_state("initial", game_state)
    gui = init_model_gui(game_id)
    render_model_gui(gui, game_id, "Model is starting.")
    llm = connect_to_llm()
    move_guard = MoveGuard()
    while True:
        if use_llm_from_start():
            move = await choose_move_with_llm(llm, game_state)
            if move is None:
                move = fallback_guess(game_state)
                move_source = "Fallback"
                print(f"{move_source} guess: {move}")
            else:
                move_source = "LLM"
                print(f"{move_source} move: {move}")
        else:
            move = deterministic_move(game_state)
            if move is not None:
                move_source = "Rule-based"
                print(f"{move_source} move: {move}")
            else:
                ranked_candidates = probability_candidates(game_state)
                if ranked_candidates:
                    best_probability = ranked_candidates[0][1]
                    tied_candidates = [
                        item for item in ranked_candidates
                        if abs(item[1] - best_probability) < 1e-9
                    ]
                    if len(tied_candidates) > 1:
                        response = await get_llm_response(
                            llm,
                            get_system_prompt(),
                            build_probability_tiebreak_prompt(game_state, tied_candidates),
                        )
                        assistant_text = response or ""
                        if assistant_text:
                            print(assistant_text)
                        try:
                            action, row, col = parse_llm_response(assistant_text)
                            move = validate_move({"action": action, "row": row, "col": col}, game_state)
                        except json.decoder.JSONDecodeError:
                            move = None

                        if move is not None and (move["x"], move["y"]) in {
                            coordinates for coordinates, _ in tied_candidates
                        }:
                            move_source = "LLM tie-break"
                            print(f"{move_source} move: {move}")
                        else:
                            move = {"action": "reveal", "x": tied_candidates[0][0][0], "y": tied_candidates[0][0][1]}
                            move_source = "Probability"
                            print(f"{move_source} move: {move}")
                    else:
                        move = {"action": "reveal", "x": ranked_candidates[0][0][0], "y": ranked_candidates[0][0][1]}
                        move_source = "Probability"
                        print(f"Probability estimate for ({move['x']}, {move['y']}): {best_probability:.3f}")
                        if DEBUG_RUN_ENABLED:
                            pretty_candidates = ", ".join(
                                f"({cx},{cy})={prob:.3f}" for (cx, cy), prob in ranked_candidates[:5]
                            )
                            print(f"[debug] top probability candidates: {pretty_candidates}")
                        print(f"{move_source} move: {move}")
                else:
                    move = await choose_move_with_llm(llm, game_state)
                    if move is None:
                        move = fallback_guess(game_state)
                        move_source = "Fallback"
                        print(f"{move_source} guess: {move}")
                    else:
                        move_source = "LLM"
                        print(f"{move_source} move: {move}")

        if move is None:
            print("No legal moves available.")
            break

        if not move_guard.accept(move, game_state):
            print(f"Blocked repeated move on an unchanged board: {move}")
            move = fallback_guess(game_state)
            move_source = "Loop-avoidance fallback"
            if move is None or not move_guard.accept(move, game_state):
                print("No non-repeating legal moves available.")
                break

        action = move["action"]
        x = move["x"]
        y = move["y"]
        move_response = make_move(game_id, action, x, y)
        if move_response is None:
            break
        game_state = get_game_state(game_id)
        debug_log_state(f"after {move_source} {action} ({x},{y})", game_state)
        if not render_model_gui(gui, game_id, f"{move_source}: {action} ({x}, {y})"):
            break
        if game_state.get("status") in ["won", "lost"]:
            print(f"Game {game_state.get('status')}!, Final Score: {game_state.get('score')}")
            debug_log_state("terminal", game_state)
            render_model_gui(gui, game_id, f"Game {game_state.get('status')}! Score {game_state.get('score')}")
            break

    if gui is not None:
        while gui["running"]:
            if not render_model_gui(gui, game_id, "Finished. Esc or close window to quit."):
                break
        pygame.quit()
        

if __name__ == "__main__":
    asyncio.run(main())
