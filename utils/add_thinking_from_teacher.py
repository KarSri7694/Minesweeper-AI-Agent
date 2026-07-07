from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable
from uuid import uuid4

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from minesweeper.training_pipeline import (  # noqa: E402
    THINKING_SFT_STAGE,
    build_user_prompt_from_state,
    load_base_system_prompt,
    make_stage_system_prompt,
    parse_completion_payload,
    validate_move,
)


def render_compact_board(board: list[list[str]]) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in board)


def _normalize_reasoning_chunk(delta: Any) -> str:
    reasoning_candidates = (
        getattr(delta, "reasoning", None),
        getattr(delta, "reasoning_content", None),
    )
    for candidate in reasoning_candidates:
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, list):
            text_parts: list[str] = []
            for item in candidate:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
            if text_parts:
                return "".join(text_parts)
    return ""


def _normalize_content_chunk(delta: Any) -> str:
    content = getattr(delta, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
        return "".join(text_parts)
    return ""


class TeacherClient:
    def stream_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> Iterable[dict[str, str]]:
        raise NotImplementedError


class OpenAICompatibleTeacherClient(TeacherClient):
    def __init__(self, *, api_key: str, base_url: str | None) -> None:
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

    def stream_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> Iterable[dict[str, str]]:
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = _normalize_reasoning_chunk(delta)
            content = _normalize_content_chunk(delta)
            yield {"reasoning": reasoning, "content": content}


class CerebrasTeacherClient(TeacherClient):
    def __init__(self, *, api_key: str) -> None:
        from cerebras.cloud.sdk import Cerebras

        self._client = Cerebras(api_key=api_key)

    def stream_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> Iterable[dict[str, str]]:
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            reasoning_effort="high",
            reasoning_format="parsed",
            stream=True,
            max_tokens=max_tokens,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = _normalize_reasoning_chunk(delta)
            content = _normalize_content_chunk(delta)
            yield {"reasoning": reasoning, "content": content}


def create_teacher_client(*, provider: str, api_key: str, base_url: str | None) -> TeacherClient:
    normalized = provider.lower()
    if normalized == "cerebras":
        return CerebrasTeacherClient(api_key=api_key)
    if normalized == "openai":
        return OpenAICompatibleTeacherClient(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    raise ValueError("provider must be either 'cerebras' or 'openai'.")


def create_remote_game(
    *,
    api_base_url: str,
    width: int,
    height: int,
    mine_density: float,
    seed: int | None,
) -> dict[str, Any]:
    response = requests.post(
        f"{api_base_url.rstrip('/')}/games",
        json={
            "width": width,
            "height": height,
            "mine_density": mine_density,
            "seed": seed,
            "output_format": "compact",
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def get_remote_state(*, api_base_url: str, game_id: str) -> dict[str, Any]:
    response = requests.post(
        f"{api_base_url.rstrip('/')}/games/{game_id}/state",
        json={"output_format": "compact"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def submit_remote_move(
    *,
    api_base_url: str,
    game_id: str,
    move: dict[str, Any],
) -> dict[str, Any]:
    response = requests.post(
        f"{api_base_url.rstrip('/')}/games/{game_id}/moves",
        json={
            "action": move["action"],
            "x": move["x"],
            "y": move["y"],
            "output_format": "compact",
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_transcript(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def maybe_launch_gui(
    *,
    launch_gui: bool,
    game_id: str,
    api_base_url: str,
) -> subprocess.Popen[str] | None:
    command = [
        sys.executable,
        str(REPO_ROOT / "src" / "game.py"),
        "--spectate-game-id",
        game_id,
        "--api-base-url",
        api_base_url,
    ]
    if not launch_gui:
        print("Spectator command:")
        print(f'$env:APP_MODE="gui"; {" ".join(command)}')
        return None

    env = os.environ.copy()
    env["APP_MODE"] = "gui"
    print("Launching pygame spectator...")
    return subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
    )


def stream_teacher_turn(
    *,
    client: TeacherClient,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> tuple[str, str]:
    thinking_chunks: list[str] = []
    content_chunks: list[str] = []

    for chunk in client.stream_completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    ):
        reasoning = chunk["reasoning"]
        content = chunk["content"]
        if reasoning:
            print(reasoning, end="", flush=True)
            thinking_chunks.append(reasoning)
        if content:
            print(content, end="", flush=True)
            content_chunks.append(content)

    print()
    return "".join(thinking_chunks), "".join(content_chunks).strip()


def build_live_teacher_row(
    *,
    session_id: str,
    turn_index: int,
    provider: str,
    model: str,
    state_before: dict[str, Any],
    state_after: dict[str, Any] | None,
    system_prompt: str,
    user_prompt: str,
    thinking_text: str,
    response_text: str,
    move: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "session_id": session_id,
        "game_id": state_before["game_id"],
        "turn_index": turn_index,
        "provider": provider,
        "teacher_model": model,
        "stage_name": THINKING_SFT_STAGE.name,
        "prompt_mode": THINKING_SFT_STAGE.prompt_mode,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "thinking_text": thinking_text,
        "response_text": response_text,
        "raw_completion": (
            f"<think>\n{thinking_text}\n</think>\n\n{response_text}"
            if thinking_text
            else response_text
        ),
        "input": str(state_before["board"]),
        "max_mines": state_before["mine_count"],
        "max_rows": state_before["height"],
        "max_columns": state_before["width"],
        "board_state_before": state_before["board"],
        "score_before": state_before["score"],
        "status_before": state_before["status"],
        "flagged_count_before": state_before["flagged_count"],
        "error": error,
    }
    if move is not None:
        row["output"] = json.dumps(move, separators=(",", ":"))
        row["action"] = move["action"]
        row["x"] = move["x"]
        row["y"] = move["y"]
    if state_after is not None:
        row["board_state_after"] = state_after["board"]
        row["score_after"] = state_after["score"]
        row["status_after"] = state_after["status"]
        row["end_reason_after"] = state_after.get("end_reason")
        row["flagged_count_after"] = state_after["flagged_count"]
        row["move_count_after"] = state_after["move_count"]
        row["last_move"] = state_after.get("last_move")
    return row


def should_persist_training_row(
    *,
    move: dict[str, Any] | None,
    state_after: dict[str, Any] | None,
    error: str | None,
) -> bool:
    if error is not None or move is None or state_after is None:
        return False

    revealed_mine = (
        move.get("action") == "reveal"
        and (
            state_after.get("end_reason") == "revealed_mine"
            or (
                state_after.get("status") == "lost"
                and (state_after.get("last_move") or {}).get("action") == "reveal"
            )
        )
    )
    return not revealed_mine


def play_live_teacher_session(
    *,
    api_base_url: str,
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None,
    output_path: Path,
    transcript_path: Path,
    width: int,
    height: int,
    mine_density: float,
    seed: int | None,
    max_turns: int,
    max_tokens: int,
    turn_delay: float,
    launch_gui: bool,
) -> dict[str, Any]:
    client = create_teacher_client(provider=provider, api_key=api_key, base_url=base_url)
    base_prompt = load_base_system_prompt()
    system_prompt = make_stage_system_prompt(base_prompt, visible_reasoning=True)
    session_id = str(uuid4())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    game_state = create_remote_game(
        api_base_url=api_base_url,
        width=width,
        height=height,
        mine_density=mine_density,
        seed=seed,
    )
    game_id = game_state["game_id"]
    maybe_launch_gui(launch_gui=launch_gui, game_id=game_id, api_base_url=api_base_url)

    append_transcript(
        transcript_path,
        (
            f"session_id: {session_id}\n"
            f"game_id: {game_id}\n"
            f"provider: {provider}\n"
            f"teacher_model: {model}\n"
            f"board: {width}x{height}\n"
            f"mine_density: {mine_density}\n"
            f"seed: {seed}\n\n"
        ),
    )

    print(f"Session ID: {session_id}")
    print(f"Game ID: {game_id}")
    print(f"Watching API session via pygame spectator for game {game_id}.")

    final_status = game_state["status"]
    final_score = game_state["score"]

    for turn_index in range(1, max_turns + 1):
        state_before = get_remote_state(api_base_url=api_base_url, game_id=game_id)
        if state_before["status"] != "in_progress":
            final_status = state_before["status"]
            final_score = state_before["score"]
            break

        user_prompt = build_user_prompt_from_state(state_before, include_score=True)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        print(f"\nTurn {turn_index}")
        print(render_compact_board(state_before["board"]))
        print(
            f"Score: {state_before['score']} | Status: {state_before['status']} "
            f"| Flags: {state_before['flagged_count']}/{state_before['mine_count']}"
        )
        print("Teacher output:")

        thinking_text = ""
        response_text = ""
        state_after = None
        move = None
        error = None

        try:
            thinking_text, response_text = stream_teacher_turn(
                client=client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            move = parse_completion_payload(
                f"<think>\n{thinking_text}\n</think>\n\n{response_text}" if thinking_text else response_text,
                prompt_mode=THINKING_SFT_STAGE.prompt_mode,
            )
            if not validate_move(state_before["height"], state_before["width"], move):
                raise ValueError("Teacher produced an out-of-bounds or invalid move.")

            state_after = submit_remote_move(
                api_base_url=api_base_url,
                game_id=game_id,
                move=move,
            )
            final_status = state_after["status"]
            final_score = state_after["score"]
            print(
                f"Applied move: {move['action']} ({move['x']}, {move['y']}) | "
                f"{state_after['last_move']['message']} | "
                f"Score: {state_after['score']} | Status: {state_after['status']}"
            )
            print(render_compact_board(state_after["board"]))
        except Exception as exc:
            error = str(exc)
            final_status = "error"
            print(f"Turn failed: {error}")

        row = build_live_teacher_row(
            session_id=session_id,
            turn_index=turn_index,
            provider=provider,
            model=model,
            state_before=state_before,
            state_after=state_after,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            thinking_text=thinking_text,
            response_text=response_text,
            move=move,
            error=error,
        )
        if should_persist_training_row(move=move, state_after=state_after, error=error):
            append_jsonl(output_path, row)
        else:
            print("Skipping dataset write for this turn due to error or mine reveal.")

        transcript = [
            f"Turn {turn_index}\n",
            f"Board before:\n{render_compact_board(state_before['board'])}\n",
            f"Score before: {state_before['score']}\n",
            f"Thinking:\n{thinking_text or '[none]'}\n",
            f"Response:\n{response_text or '[none]'}\n",
            f"Parsed move: {json.dumps(move) if move is not None else '[none]'}\n",
        ]
        if state_after is not None:
            transcript.append(f"Board after:\n{render_compact_board(state_after['board'])}\n")
            transcript.append(
                f"Result: {state_after['last_move']['message']} | "
                f"Score delta: {state_after['last_move']['score_delta']:+d}\n"
            )
            transcript.append(
                f"Score after: {state_after['score']} | Status after: {state_after['status']}\n"
            )
        if error is not None:
            transcript.append(f"Error: {error}\n")
        transcript.append("-" * 60 + "\n")
        append_transcript(transcript_path, "".join(transcript))

        if error is not None:
            break
        if state_after is not None and state_after["status"] != "in_progress":
            break
        if turn_delay > 0:
            time.sleep(turn_delay)

    summary = {
        "session_id": session_id,
        "game_id": game_id,
        "final_status": final_status,
        "final_score": final_score,
        "dataset_path": str(output_path),
        "transcript_path": str(transcript_path),
    }
    append_transcript(
        transcript_path,
        f"Final status: {final_status}\nFinal score: {final_score}\n",
    )
    return summary


def play_live_teacher_sessions(
    *,
    num_games: int,
    api_base_url: str,
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None,
    output_path: Path,
    transcript_path: Path,
    width: int,
    height: int,
    mine_density: float,
    seed: int | None,
    max_turns: int,
    max_tokens: int,
    turn_delay: float,
    launch_gui: bool,
) -> list[dict[str, Any]]:
    if num_games < 1:
        raise ValueError("--num-games must be at least 1.")

    summaries: list[dict[str, Any]] = []
    base_seed = seed if seed is not None else None

    for game_index in range(num_games):
        session_seed = None if base_seed is None else base_seed + game_index
        print(f"\n=== Starting game {game_index + 1}/{num_games} ===")
        summary = play_live_teacher_session(
            api_base_url=api_base_url,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            output_path=output_path,
            transcript_path=transcript_path,
            width=width,
            height=height,
            mine_density=mine_density,
            seed=session_seed,
            max_turns=max_turns,
            max_tokens=max_tokens,
            turn_delay=turn_delay,
            launch_gui=launch_gui,
        )
        summary["game_index"] = game_index + 1
        summaries.append(summary)

    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play a live Minesweeper game with a teacher model while saving streamed thinking data.",
    )
    parser.add_argument("--provider", choices=["cerebras", "openai"], default="cerebras")
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--game-api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--mine-density", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-games", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=80000)
    parser.add_argument("--turn-delay", type=float, default=0.75)
    parser.add_argument("--output-jsonl", default="dataset/live_teacher_thinking.jsonl")
    parser.add_argument("--transcript-path", default="dataset/live_teacher_session.log")
    parser.add_argument("--launch-gui", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    api_key = args.api_key
    if not api_key:
        env_var = "CEREBRAS_API_KEY" if args.provider == "cerebras" else "OPENROUTER_API_KEY"
        api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError("No API key provided. Pass --api-key or set the provider-specific environment variable.")

    summaries = play_live_teacher_sessions(
        num_games=args.num_games,
        api_base_url=args.game_api_base_url,
        provider=args.provider,
        model=args.teacher_model,
        api_key=api_key,
        base_url=args.base_url,
        output_path=(REPO_ROOT / args.output_jsonl).resolve(),
        transcript_path=(REPO_ROOT / args.transcript_path).resolve(),
        width=args.width,
        height=args.height,
        mine_density=args.mine_density,
        seed=args.seed,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        turn_delay=args.turn_delay,
        launch_gui=args.launch_gui,
    )
    print("\nAll sessions complete.")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
