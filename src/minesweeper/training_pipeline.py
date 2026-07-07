from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from datasets import Dataset

from .engine import GameConfig, GameEngine
from .rl_rewards import calculate_reward_components

MovePayload = dict[str, Any]
TeacherRewriter = Callable[[dict[str, Any]], str]
CompletionGenerator = Callable[[str], str]

MALFORMED_COMPLETION_PENALTY = -1.0
SNAPSHOT_LOAD_PENALTY = -1.0
INVALID_MOVE_PENALTY = -0.8
EXECUTION_ERROR_PENALTY = -0.66


@dataclass(frozen=True)
class StageSpec:
    name: str
    prompt_mode: str
    trainer_kind: str
    visible_reasoning: bool


THINKING_SFT_STAGE = StageSpec(
    name="thinking-sft",
    prompt_mode="thinking",
    trainer_kind="sft",
    visible_reasoning=True,
)
THINKING_GRPO_STAGE = StageSpec(
    name="thinking-grpo",
    prompt_mode="thinking",
    trainer_kind="grpo",
    visible_reasoning=True,
)
DIRECT_SFT_STAGE = StageSpec(
    name="direct-sft",
    prompt_mode="direct",
    trainer_kind="sft",
    visible_reasoning=False,
)
DIRECT_GRPO_STAGE = StageSpec(
    name="direct-grpo",
    prompt_mode="direct",
    trainer_kind="grpo",
    visible_reasoning=False,
)


def load_base_system_prompt(path: str | Path | None = None) -> str:
    prompt_path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "system_prompt.md"
    return prompt_path.read_text(encoding="utf-8").strip()


def make_stage_system_prompt(base_prompt: str, *, visible_reasoning: bool) -> str:
    if visible_reasoning:
        suffix = (
            "\n\nReasoning protocol:\n"
            "1. Think through the visible constraints before deciding.\n"
            "2. Put the reasoning inside <think>...</think>.\n"
            "3. After the reasoning block, output exactly one JSON move payload.\n"
            "4. The final answer must still be one legal move."
        )
    else:
        suffix = (
            "\n\nResponse protocol:\n"
            "Output exactly one JSON move payload and do not reveal reasoning.\n"
            "Do not emit <think> tags or explanatory text."
        )
    return f"{base_prompt.strip()}{suffix}"


def build_user_prompt_from_state(state: dict[str, Any], *, include_score: bool = False) -> str:
    lines = [
        f"Board State: {state['board']}",
        f"Max Mines: {state['mine_count']}",
        f"Max Rows: {state['height']}",
        f"Max Columns: {state['width']}",
    ]
    if include_score:
        lines.insert(1, f"Score: {state['score']}")
    return "\n".join(lines)


def _parse_board(board_value: Any) -> list[list[str]]:
    if isinstance(board_value, str):
        return ast.literal_eval(board_value)
    return board_value


def _parse_action(action_value: Any) -> dict[str, Any]:
    if isinstance(action_value, str):
        return json.loads(action_value)
    return action_value


def _neighbor_coordinates(board: Sequence[Sequence[str]], x: int, y: int) -> Iterable[tuple[int, int]]:
    height = len(board)
    width = len(board[0]) if height else 0
    for ny in range(max(0, y - 1), min(height, y + 2)):
        for nx in range(max(0, x - 1), min(width, x + 2)):
            if nx == x and ny == y:
                continue
            yield nx, ny


def summarize_local_constraints(board_value: Any, x: int, y: int) -> list[str]:
    board = _parse_board(board_value)
    lines: list[str] = []
    for nx, ny in _neighbor_coordinates(board, x, y):
        tile = board[ny][nx]
        if not str(tile).isdigit():
            continue

        hidden_neighbors = 0
        flagged_neighbors = 0
        for adj_x, adj_y in _neighbor_coordinates(board, nx, ny):
            neighbor_value = board[adj_y][adj_x]
            if neighbor_value == ".":
                hidden_neighbors += 1
            elif neighbor_value == "F":
                flagged_neighbors += 1

        lines.append(
            f"Cell ({nx}, {ny}) shows {tile} with {flagged_neighbors} flagged and {hidden_neighbors} hidden neighbors."
        )

    return lines[:4]


def build_programmatic_reasoning_trace(row: dict[str, Any]) -> str:
    board = _parse_board(row["input"])
    action = _parse_action(row["output"])
    x = int(action["x"])
    y = int(action["y"])
    move_type = row.get("move_type", "probabilistic")
    candidate_rank = int(row.get("candidate_rank", 1))
    candidate_count = int(row.get("candidate_count", 1))
    mine_probability = float(row.get("mine_probability_at_action", 0.0))

    lines = ["Read the full visible board and focus on the frontier around the target tile."]
    lines.extend(summarize_local_constraints(board, x, y))

    if move_type == "deterministic":
        if action["action"] == "flag":
            lines.append(
                f"The move is deterministic: tile ({x}, {y}) must be a mine because the nearby numbered constraints are saturated."
            )
        else:
            lines.append(
                f"The move is deterministic: tile ({x}, {y}) is forced safe after accounting for the nearby flags and numbered constraints."
            )
    else:
        lines.append("No deterministic move is available on this board state.")
        lines.append(
            f"Choose the ranked candidate {candidate_rank} of {candidate_count} with estimated mine probability {mine_probability:.3f}."
        )
        lines.append(
            f"Because the action is a guess, prefer revealing the lowest-risk tile instead of placing an uncertain flag."
        )

    lines.append(f"Return the move {json.dumps(action, separators=(',', ':'))}.")
    return "\n".join(lines)


def build_reasoning_target(
    row: dict[str, Any],
    *,
    use_teacher_rewrite: bool = False,
    teacher_rewriter: TeacherRewriter | None = None,
) -> str:
    if use_teacher_rewrite and teacher_rewriter is not None:
        reasoning = teacher_rewriter(row)
    else:
        reasoning = build_programmatic_reasoning_trace(row)

    action = _parse_action(row["output"])
    action_json = json.dumps(action, separators=(",", ":"))
    return f"<think>\n{reasoning}\n</think>\n\n{action_json}"


def live_teacher_row_to_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    output = row.get("output")
    if output is None and row.get("action") is not None:
        output = json.dumps(
            {
                "action": row["action"],
                "x": row["x"],
                "y": row["y"],
            },
            separators=(",", ":"),
        )

    return {
        "input": row["input"],
        "output": output,
        "thinking": row.get("thinking_text"),
        "max_mines": row["max_mines"],
        "max_rows": row["max_rows"],
        "max_columns": row["max_columns"],
        "snapshot": row.get("snapshot"),
        "session_id": row.get("session_id"),
        "game_id": row.get("game_id"),
        "turn_index": row.get("turn_index"),
    }


def _build_messages(
    row: dict[str, Any],
    *,
    system_prompt: str,
    assistant_content: str | None,
) -> list[dict[str, str]]:
    user_content = (
        f"Board State: {row['input']}\n"
        f"Max Mines: {row['max_mines']}\n"
        f"Max Rows: {row['max_rows']}\n"
        f"Max Columns: {row['max_columns']}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


def prepare_sft_dataframe(
    frame,
    *,
    stage: StageSpec,
    system_prompt: str,
    use_teacher_rewrite: bool = False,
    teacher_rewriter: TeacherRewriter | None = None,
):
    formatted = frame.copy()
    assistant_builder = (
        lambda row: build_reasoning_target(
            row,
            use_teacher_rewrite=use_teacher_rewrite,
            teacher_rewriter=teacher_rewriter,
        )
        if stage.visible_reasoning
        else json.dumps(_parse_action(row["output"]), separators=(",", ":"))
    )

    formatted["Messages"] = formatted.apply(
        lambda row: _build_messages(
            row,
            system_prompt=system_prompt,
            assistant_content=assistant_builder(row),
        ),
        axis=1,
    )
    formatted["EvalMessages"] = formatted.apply(
        lambda row: _build_messages(
            row,
            system_prompt=system_prompt,
            assistant_content=None,
        ),
        axis=1,
    )
    return formatted


def dataframe_to_hf_dataset(frame, tokenizer, *, message_column: str = "Messages") -> Dataset:
    converted = frame.copy()
    converted["text"] = tokenizer.apply_chat_template(
        converted[message_column].tolist(),
        tokenize=False,
        add_generation_prompt=False,
    )
    return Dataset.from_pandas(converted, preserve_index=False)


def sample_prefill_move(
    game: GameEngine,
    rng: random.Random,
    *,
    reveal_probability: float = 0.7,
) -> dict[str, Any] | None:
    hidden_tiles = [tile for tile in game.iter_tiles() if not tile.is_revealed and not tile.is_flagged]
    if not hidden_tiles:
        return None
    if rng.random() < reveal_probability:
        tile = rng.choice(hidden_tiles)
        return {"action": "reveal", "x": tile.x, "y": tile.y}
    tile = rng.choice(hidden_tiles)
    return {"action": "flag", "x": tile.x, "y": tile.y}


def create_live_game(
    *,
    width: int,
    height: int,
    mine_density: float,
    seed: int | None = None,
    max_prefill_moves: int = 4,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    rng = random.Random(seed) if seed is not None else random.Random()
    game = GameEngine(config=GameConfig(width=width, height=height, mine_density=mine_density), rng=rng)

    prefill_moves = rng.randint(0, max_prefill_moves)
    if prefill_moves > 0:
        first_x = rng.randrange(width)
        first_y = rng.randrange(height)
        game.reveal(first_x, first_y)
        for _ in range(prefill_moves - 1):
            if game.status.value != "in_progress":
                break
            move = sample_prefill_move(game, rng)
            if move is None:
                break
            try:
                if move["action"] == "reveal":
                    game.reveal(move["x"], move["y"])
                else:
                    game.flag(move["x"], move["y"])
            except ValueError:
                continue

    visible_state = game.compact_state()
    full_board_state = game.full_board_compact_state() if game.snapshot()["mines_placed"] else None
    return visible_state, full_board_state, game.snapshot()


def build_live_dataset(
    tokenizer,
    *,
    system_prompt: str,
    stage: StageSpec,
    num_examples: int,
    board_sizes: Sequence[tuple[int, int]],
    mine_densities: Sequence[float],
    seed: int | None = None,
) -> Dataset:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    while len(rows) < num_examples:
        width, height = rng.choice(list(board_sizes))
        density = rng.choice(list(mine_densities))
        visible_state, full_state, snapshot = create_live_game(
            width=width,
            height=height,
            mine_density=density,
            seed=rng.randrange(0, 2**31),
        )
        if visible_state["status"] != "in_progress":
            continue

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt_from_state(visible_state)},
        ]
        rows.append(
            {
                "prompt": tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "rows": visible_state["height"],
                "columns": visible_state["width"],
                "max_mines": visible_state["mine_count"],
                "board_state": str(visible_state["board"]),
                "revealed_board": str(full_state),
                "snapshot": str(snapshot),
                "stage_name": stage.name,
                "prompt_mode": stage.prompt_mode,
            }
        )

    return Dataset.from_list(rows)


def extract_last_json_object(text: str) -> str:
    end = text.rfind("}")
    if end == -1:
        raise ValueError("No JSON object found in completion.")

    depth = 0
    in_string = False
    escape = False
    start = None
    for index in range(end, -1, -1):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "}":
            depth += 1
        elif char == "{":
            depth -= 1
            if depth == 0:
                start = index
                break

    if start is None:
        raise ValueError("Could not isolate the final JSON object.")
    return text[start : end + 1]


def parse_completion_payload(completion: str, *, prompt_mode: str) -> MovePayload:
    if not isinstance(completion, str):
        raise TypeError("Completion must be a string.")
    payload_text = extract_last_json_object(completion.strip())
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise TypeError("Completion payload must decode to a dict.")
    nested_action = payload.get("action")
    if isinstance(nested_action, dict):
        if {"action", "x", "y"}.issubset(nested_action):
            payload = nested_action
        elif isinstance(nested_action.get("type"), str):
            payload = dict(payload)
            payload["action"] = nested_action["type"]
    if prompt_mode == "direct" and "<think>" in completion:
        raise ValueError("Direct mode completion leaked visible reasoning.")
    return payload


def validate_move(rows: int, columns: int, response: MovePayload) -> bool:
    if not isinstance(response, dict):
        return False
    action = response.get("action")
    x = response.get("x")
    y = response.get("y")
    return (
        isinstance(action, str)
        and action in {"reveal", "flag"}
        and isinstance(x, int)
        and isinstance(y, int)
        and 0 <= x < columns
        and 0 <= y < rows
    )


def build_game_from_snapshot(snapshot: str | dict[str, Any]) -> GameEngine:
    snapshot_dict = ast.literal_eval(snapshot) if isinstance(snapshot, str) else snapshot
    return GameEngine.from_snapshot(snapshot_dict)


def build_reward_function(stage: StageSpec):
    def score_completion_reward(i: int, completion: str, rows_list, columns_list, snapshots):
        rows = rows_list[i] if i < len(rows_list) else None
        columns = columns_list[i] if i < len(columns_list) else None
        snapshot = snapshots[i] if i < len(snapshots) else None
        if snapshot is None or rows is None or columns is None:
            raise ValueError(f"Missing required snapshot information for reward calculation at index {i}")

        try:
            response = parse_completion_payload(completion, prompt_mode=stage.prompt_mode)
        except Exception:
            return {
                "reward": float(MALFORMED_COMPLETION_PENALTY),
                "component": "malformed_completion",
            }

        try:
            game = build_game_from_snapshot(snapshot)
        except Exception:
            return {
                "reward": float(SNAPSHOT_LOAD_PENALTY),
                "component": "snapshot_load_failure",
            }

        if not validate_move(rows, columns, response):
            return {
                "reward": float(INVALID_MOVE_PENALTY),
                "component": "invalid_move",
            }

        try:
            reward_components = calculate_reward_components(game, response)
        except ValueError as exc:
            if "Revealed tiles cannot be flagged." in str(exc):
                return {
                    "reward": float(-0.5),
                    "component": "revealed_tiles_cannot_be_flagged",
                }
            return {
                "reward": float(EXECUTION_ERROR_PENALTY),
                "component": "execution_error",
            }

        return {
            "reward": float(reward_components["total_reward"]),
            "component": "move_reward",
            "reward_components": reward_components,
        }

    def calculate_reward(prompts=None, completions=None, **kwargs):
        rows_list = kwargs.get("rows") or []
        columns_list = kwargs.get("columns") or []
        snapshots = kwargs.get("snapshot") or []

        rewards = []
        reward_logs = []
        for i, completion in enumerate(completions or []):
            reward_result = score_completion_reward(i, completion, rows_list, columns_list, snapshots)
            rewards.append(reward_result["reward"])
            reward_logs.append(reward_result)

        calculate_reward.last_logs = reward_logs
        return rewards

    return calculate_reward


def generate_text_from_model(
    model,
    tokenizer,
    messages: Sequence[dict[str, str]],
    *,
    max_new_tokens: int = 256,
    do_sample: bool = False,
) -> str:
    text = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    stop_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id and im_end_id not in stop_ids:
        stop_ids.append(im_end_id)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=stop_ids,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()
    for stop_token in ("<|im_end|>", "<|endoftext|>"):
        if stop_token in generated_text:
            generated_text = generated_text.split(stop_token, 1)[0].strip()
    return generated_text


def build_distillation_dataset(
    source_rows: Sequence[dict[str, Any]],
    *,
    system_prompt: str,
    tokenizer,
    generate_completion: CompletionGenerator,
    max_samples_per_prompt: int = 1,
) -> Dataset:
    distilled_rows: list[dict[str, Any]] = []

    for row in source_rows:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_user_prompt_from_state(
                    {
                        "board": _parse_board(row["input"]),
                        "mine_count": row["max_mines"],
                        "height": row["max_rows"],
                        "width": row["max_columns"],
                        "score": row.get("score", 0),
                    }
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        best_row: dict[str, Any] | None = None
        for _ in range(max_samples_per_prompt):
            completion = generate_completion(prompt)
            try:
                move = parse_completion_payload(completion, prompt_mode="thinking")
            except Exception:
                continue

            if not validate_move(int(row["max_rows"]), int(row["max_columns"]), move):
                continue

            game = build_game_from_snapshot(row["snapshot"]) if "snapshot" in row else None
            reward = None
            if game is not None:
                reward = float(calculate_reward_components(game, move)["total_reward"])

            candidate = {
                "input": row["input"],
                "output": json.dumps(move, separators=(",", ":")),
                "teacher_completion": completion,
                "teacher_reward": reward,
                "max_mines": row["max_mines"],
                "max_rows": row["max_rows"],
                "max_columns": row["max_columns"],
            }
            if best_row is None or (reward is not None and reward > (best_row["teacher_reward"] or float("-inf"))):
                best_row = candidate

        if best_row is not None:
            distilled_rows.append(best_row)

    return Dataset.from_list(distilled_rows)


def summarize_evaluation_records(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}

    games = float(len(records))
    return {
        "games": games,
        "win_rate": sum(1.0 for record in records if record["status"] == "won") / games,
        "avg_score": sum(float(record["score"]) for record in records) / games,
        "avg_turns": sum(float(record["turns"]) for record in records) / games,
        "valid_move_rate": sum(float(record["valid_move_rate"]) for record in records) / games,
        "json_rate": sum(float(record["json_rate"]) for record in records) / games,
    }


def evaluate_policy(
    *,
    model,
    tokenizer,
    system_prompt: str,
    stage: StageSpec,
    board_sizes: Sequence[tuple[int, int]],
    mine_densities: Sequence[float],
    games_per_setting: int = 4,
    seed: int = 42,
    max_turns: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for width, height in board_sizes:
        for density in mine_densities:
            for _ in range(games_per_setting):
                game = GameEngine(
                    config=GameConfig(width=width, height=height, mine_density=density),
                    rng=random.Random(rng.randrange(0, 2**31)),
                )
                turns = 0
                valid_moves = 0
                json_moves = 0

                while game.status.value == "in_progress" and turns < max_turns:
                    state = game.compact_state()
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": build_user_prompt_from_state(state, include_score=True)},
                    ]
                    completion = generate_text_from_model(model, tokenizer, messages, max_new_tokens=256, do_sample=False)
                    try:
                        move = parse_completion_payload(completion, prompt_mode=stage.prompt_mode)
                        json_moves += 1
                    except Exception:
                        turns += 1
                        continue

                    if not validate_move(state["height"], state["width"], move):
                        turns += 1
                        continue

                    valid_moves += 1
                    if move["action"] == "reveal":
                        game.reveal(move["x"], move["y"])
                    else:
                        game.flag(move["x"], move["y"])
                    turns += 1

                records.append(
                    {
                        "width": width,
                        "height": height,
                        "mine_density": density,
                        "status": game.status.value,
                        "score": game.score,
                        "turns": turns,
                        "json_rate": json_moves / max(turns, 1),
                        "valid_move_rate": valid_moves / max(turns, 1),
                    }
                )

    return records, summarize_evaluation_records(records)
