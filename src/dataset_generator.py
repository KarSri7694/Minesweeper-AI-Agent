"""Generate logic-labelled Minesweeper examples for SFT or offline analysis.

The generator uses the production game engine and policy helpers, so its board
encoding and actions stay aligned with what an agent sees at inference time.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from llm_parser import deterministic_move, fallback_guess, probability_candidates
from minesweeper.engine import GameConfig, GameEngine, GameStatus


def select_oracle_move(state: dict) -> tuple[dict | None, str]:
    move = deterministic_move(state)
    if move is not None:
        return move, "logical"
    candidates = probability_candidates(state)
    if candidates:
        (x, y), probability = candidates[0]
        return {"action": "reveal", "x": x, "y": y}, f"minimum_risk:{probability:.4f}"
    return fallback_guess(state), "fallback"


def generate_examples(
    samples: int,
    width: int,
    height: int,
    mine_density: float,
    seed: int,
) -> list[dict]:
    """Return labelled states, prioritising moves justified by visible constraints."""
    rng = random.Random(seed)
    examples: list[dict] = []
    game_index = 0
    while len(examples) < samples:
        game = GameEngine(
            config=GameConfig(width=width, height=height, mine_density=mine_density),
            rng=random.Random(rng.randrange(2**63)),
        )
        game_index += 1
        while game.status is GameStatus.IN_PROGRESS and len(examples) < samples:
            state = game.compact_state()
            move, policy = select_oracle_move(state)
            if move is None:
                break
            # The first reveal has no logical evidence; retain the example only
            # when callers specifically need opening moves.
            if state["move_count"] > 0 and policy == "logical":
                examples.append(
                    {
                        "board": state["board"],
                        "width": state["width"],
                        "height": state["height"],
                        "mine_count": state["mine_count"],
                        "flagged_count": state["flagged_count"],
                        "move_count": state["move_count"],
                        "action": move,
                        "label": policy,
                        "game_index": game_index,
                    }
                )
            if move["action"] == "reveal":
                game.reveal(move["x"], move["y"])
            else:
                game.flag(move["x"], move["y"])
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate logic-labelled Minesweeper JSONL data.")
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--mine-density", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("datasets/logic_moves.jsonl"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    examples = generate_examples(
        args.samples, args.width, args.height, args.mine_density, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for example in examples:
            output.write(json.dumps(example, separators=(",", ":")) + "\n")
    print(f"Wrote {len(examples)} logic-labelled examples to {args.output}")


if __name__ == "__main__":
    main()
