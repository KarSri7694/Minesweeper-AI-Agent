from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_generator import build_transition_rows
from minesweeper.solve_algo import (
    Board,
    FLAGGED,
    MINE,
    ProbabilisticSolver,
    REVEALED,
    ranked_solver_actions,
)


def make_board_with_two_safe_reveals() -> Board:
    grid = np.zeros((2, 2), dtype=int)
    board = Board.from_config(grid, {(0, 0)})
    board.grid[0, 0] = MINE
    board._calculate_numbers()
    board.state[1, 1] = REVEALED
    board.state[0, 0] = FLAGGED
    return board


def make_board_with_three_forced_flags() -> Board:
    grid = np.zeros((2, 2), dtype=int)
    board = Board.from_config(grid, {(0, 0), (0, 1), (1, 0)})
    board.grid[0, 0] = MINE
    board.grid[0, 1] = MINE
    board.grid[1, 0] = MINE
    board._calculate_numbers()
    board.state[1, 1] = REVEALED
    return board


class DatasetGenerationTests(unittest.TestCase):
    def test_ranked_solver_actions_returns_all_deterministic_reveals(self) -> None:
        board = make_board_with_two_safe_reveals()

        candidates = ranked_solver_actions(board)

        self.assertEqual(
            [(candidate.action, candidate.row, candidate.col) for candidate in candidates],
            [("reveal", 0, 1), ("reveal", 1, 0)],
        )
        self.assertTrue(all(candidate.move_type == "deterministic" for candidate in candidates))

    def test_ranked_solver_actions_returns_all_deterministic_flags(self) -> None:
        board = make_board_with_three_forced_flags()

        candidates = ranked_solver_actions(board)

        self.assertEqual(
            [(candidate.action, candidate.row, candidate.col) for candidate in candidates],
            [("flag", 0, 0), ("flag", 0, 1), ("flag", 1, 0)],
        )
        self.assertTrue(all(candidate.move_type == "deterministic" for candidate in candidates))

    def test_ranked_solver_actions_limits_probabilistic_outputs_to_top_three(self) -> None:
        board = Board(3, 3, 2)
        board.first_click = False
        probabilities = {
            (0, 0): 0.40,
            (0, 1): 0.10,
            (1, 0): 0.10,
            (1, 1): 0.10,
            (2, 2): 0.30,
        }

        with patch.object(ProbabilisticSolver, "calculate_probabilities", return_value=probabilities):
            candidates = ranked_solver_actions(board)

        self.assertEqual(
            [(candidate.row, candidate.col) for candidate in candidates],
            [(1, 1), (0, 1), (1, 0)],
        )
        self.assertTrue(all(candidate.action == "reveal" for candidate in candidates))
        self.assertTrue(all(candidate.move_type == "probabilistic" for candidate in candidates))

    def test_build_transition_rows_duplicates_board_state_for_each_candidate(self) -> None:
        board = make_board_with_two_safe_reveals()
        candidates = ranked_solver_actions(board)

        rows = build_transition_rows(board, candidates, mine_count=1, rows=2, cols=2)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["board_state_before"], rows[1]["board_state_before"])
        self.assertNotEqual(rows[0]["action"], rows[1]["action"])
        self.assertEqual(rows[0]["candidate_rank"], 1)
        self.assertEqual(rows[1]["candidate_rank"], 2)
        self.assertEqual(rows[0]["candidate_count"], 2)
        self.assertEqual(rows[1]["candidate_count"], 2)

        action_1 = json.loads(rows[0]["action"])
        action_2 = json.loads(rows[1]["action"])
        after_1 = json.loads(rows[0]["board_state_after"])
        after_2 = json.loads(rows[1]["board_state_after"])

        self.assertEqual(action_1, {"action": "reveal", "x": 1, "y": 0})
        self.assertEqual(action_2, {"action": "reveal", "x": 0, "y": 1})
        self.assertEqual(after_1[0][1], "1")
        self.assertEqual(after_1[1][0], ".")
        self.assertEqual(after_2[1][0], "1")
        self.assertEqual(after_2[0][1], ".")


if __name__ == "__main__":
    unittest.main()
