"""Reproducible offline benchmark for the hybrid Minesweeper policy."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_parser import deterministic_move, fallback_guess, probability_candidates
from minesweeper.engine import GameConfig, GameEngine, GameStatus


def choose_move(state: dict) -> tuple[dict | None, str]:
    move = deterministic_move(state)
    if move:
        return move, "logical"
    candidates = probability_candidates(state)
    if candidates:
        (x, y), _ = candidates[0]
        return {"action": "reveal", "x": x, "y": y}, "minimum_risk"
    return fallback_guess(state), "fallback"


def run_game(width: int, height: int, mine_density: float, seed: int) -> dict:
    game = GameEngine(GameConfig(width, height, mine_density), rng=random.Random(seed))
    moves = safe_reveals = correct_flags = wrong_flags = 0
    policy_counts: dict[str, int] = {}
    while game.status is GameStatus.IN_PROGRESS:
        move, source = choose_move(game.compact_state())
        if move is None:
            break
        policy_counts[source] = policy_counts.get(source, 0) + 1
        tile = game.get_tile(move["x"], move["y"])
        if move["action"] == "reveal":
            if not tile.is_mine:
                safe_reveals += 1
            game.reveal(move["x"], move["y"])
        else:
            if tile.is_mine:
                correct_flags += 1
            else:
                wrong_flags += 1
            game.flag(move["x"], move["y"])
        moves += 1
    return {
        "won": game.status is GameStatus.WON,
        "moves": moves,
        "safe_reveals": safe_reveals,
        "correct_flags": correct_flags,
        "wrong_flags": wrong_flags,
        "score": game.score,
        "policy_counts": policy_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the deterministic/probability policy.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--mine-density", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [run_game(args.width, args.height, args.mine_density, args.seed + index) for index in range(args.games)]
    total = lambda key: sum(result[key] for result in results)
    flags = total("correct_flags") + total("wrong_flags")
    report = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "games": args.games,
        "win_rate": total("won") / args.games,
        "average_moves": total("moves") / args.games,
        "average_score": total("score") / args.games,
        "safe_reveals": total("safe_reveals"),
        "flag_precision": total("correct_flags") / flags if flags else None,
        "wrong_flags": total("wrong_flags"),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
