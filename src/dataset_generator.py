from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from minesweeper.solve_algo import Board, CandidateAction, apply_candidate_action, ranked_solver_actions


def build_transition_rows(
    board: Board,
    candidates: list[CandidateAction],
    mine_count: int,
    rows: int,
    cols: int,
) -> list[dict]:
    before_state = board.compact_state()
    candidate_count = len(candidates)
    transitions: list[dict] = []

    for rank, candidate in enumerate(candidates, start=1):
        candidate_board = board.clone()
        apply_candidate_action(candidate_board, candidate)
        after_state = candidate_board.compact_state()

        transitions.append(
            {
                "board_state_before": json.dumps(before_state["board"]),
                "board_state_after": json.dumps(after_state["board"]),
                "score": after_state["score"],
                "action": json.dumps({"action": candidate.action, "x": candidate.col, "y": candidate.row}),
                "max_mines": mine_count,
                "max_rows": rows,
                "max_columns": cols,
                "candidate_rank": rank,
                "candidate_count": candidate_count,
                "move_type": candidate.move_type,
                "mine_probability_at_action": candidate.mine_probability,
            }
        )

    return transitions


def play_full_game(rows: int, cols: int, mine_density: float, seed: int) -> list[dict]:
    total_cells = rows * cols
    mine_count = int(total_cells * mine_density)
    board = Board(rows, cols, mine_count)

    transitions: list[dict] = []

    while not board.game_over:
        candidates = ranked_solver_actions(board)
        if not candidates:
            raise RuntimeError("Solver could not find a deterministic or probabilistic move.")

        transitions.extend(build_transition_rows(board, candidates, mine_count, rows, cols))
        apply_candidate_action(board, candidates[0])

    return transitions


def generate_dataset(
    output_path: str,
    games: int,
    rows: int,
    cols: int,
    mine_density: float,
    seed: int | None,
) -> Path:
    if not 0.0 <= mine_density < 1.0:
        raise ValueError("--mine-density must be in the range [0, 1).")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        random.seed(seed)

    fieldnames = [
        "board_state_before",
        "board_state_after",
        "score",
        "action",
        "max_mines",
        "max_rows",
        "max_columns",
        "candidate_rank",
        "candidate_count",
        "move_type",
        "mine_probability_at_action",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for game_index in range(games):
            game_seed = (seed + game_index) if seed is not None else random.randrange(0, 2**31)
            random.seed(game_seed)
            transitions = play_full_game(
                rows=rows,
                cols=cols,
                mine_density=mine_density,
                seed=game_seed,
            )
            writer.writerows(transitions)

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a CSV dataset of complete fresh-start Minesweeper games using solve_algo."
    )
    parser.add_argument("--output", default="full_games_dataset.csv")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--mine-density", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_file = generate_dataset(
        output_path=args.output,
        games=args.games,
        rows=args.rows,
        cols=args.cols,
        mine_density=args.mine_density,
        seed=args.seed,
    )
    print(f"Saved full-game dataset to {output_file.resolve()}")


if __name__ == "__main__":
    main()
