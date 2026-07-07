from __future__ import annotations

import random
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minesweeper.engine import GameConfig, GameEngine, GameStatus
from minesweeper.rl_rewards import (
    CORRECT_FLAG_REWARD,
    DETERMINISTIC_MATCH_BONUS,
    FRONTIER_PROGRESS_BONUS,
    INVALID_FLAG_PENALTY,
    INVALID_REPEAT_FLAG_PENALTY,
    INVALID_REVEAL_PENALTY,
    MISSED_DETERMINISTIC_PENALTY,
    MINE_REVEAL_PENALTY,
    NUMBER_REVEAL_REWARD,
    PROBABILISTIC_FLAG_PENALTY,
    WIN_REWARD,
    WRONG_FLAG_PENALTY,
    ZERO_REVEAL_REWARD,
    apply_move_and_get_reward,
    calculate_reward_components,
)


def build_game_from_layout(
    width: int,
    height: int,
    mines: set[tuple[int, int]],
    *,
    revealed: set[tuple[int, int]] | None = None,
    flagged: set[tuple[int, int]] | None = None,
) -> GameEngine:
    revealed = revealed or set()
    flagged = flagged or set()

    game = GameEngine(config=GameConfig(width=width, height=height, mine_density=0.2))
    snapshot = game.snapshot()
    snapshot["mines_placed"] = True
    snapshot["rng_state"] = random.Random(0).getstate()

    for y, row in enumerate(snapshot["board"]):
        for x, tile in enumerate(row):
            is_mine = (x, y) in mines
            is_flagged = (x, y) in flagged
            tile["is_mine"] = is_mine
            tile["adjacent_mines"] = 0
            tile["is_revealed"] = (x, y) in revealed
            tile["is_flagged"] = is_flagged
            tile["flag_score_applied"] = is_flagged

    for y, row in enumerate(snapshot["board"]):
        for x, tile in enumerate(row):
            if tile["is_mine"]:
                continue
            adjacent = 0
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if (nx, ny) != (x, y) and snapshot["board"][ny][nx]["is_mine"]:
                        adjacent += 1
            tile["adjacent_mines"] = adjacent

    return GameEngine.from_snapshot(snapshot)


class RlRewardTests(unittest.TestCase):
    def test_revealing_numbered_tile_returns_base_reward(self) -> None:
        game = build_game_from_layout(width=3, height=3, mines={(0, 0)})

        reward = calculate_reward_components(game, {"action": "reveal", "x": 1, "y": 0})

        self.assertEqual(reward["base_reward"], NUMBER_REVEAL_REWARD)
        self.assertEqual(reward["logic_state"], "probabilistic")
        self.assertEqual(reward["logic_match"], "unranked_guess")
        self.assertGreater(reward["total_reward"], NUMBER_REVEAL_REWARD)
        self.assertTrue(game.get_tile(1, 0).is_revealed)

    def test_zero_reveal_returns_fixed_reward_not_cascade_size(self) -> None:
        game = build_game_from_layout(width=3, height=3, mines={(0, 0)})

        reward = calculate_reward_components(game, {"action": "reveal", "x": 2, "y": 2})

        self.assertEqual(reward["base_reward"], ZERO_REVEAL_REWARD)
        self.assertEqual(reward["logic_state"], "probabilistic")
        self.assertGreater(reward["total_reward"], ZERO_REVEAL_REWARD)
        self.assertGreater(sum(tile.is_revealed for tile in game.iter_tiles()), 1)

    def test_revealing_mine_returns_large_penalty(self) -> None:
        game = build_game_from_layout(width=3, height=3, mines={(0, 0)})

        reward = calculate_reward_components(game, {"action": "reveal", "x": 0, "y": 0})

        self.assertEqual(reward["base_reward"], MINE_REVEAL_PENALTY)
        self.assertEqual(reward["logic_state"], "probabilistic")
        self.assertGreater(reward["total_reward"], MINE_REVEAL_PENALTY)
        self.assertIs(game.status, GameStatus.LOST)

    def test_correct_flag_returns_bonus(self) -> None:
        game = build_game_from_layout(width=3, height=3, mines={(0, 0)})

        reward = calculate_reward_components(game, {"action": "flag", "x": 0, "y": 0})

        self.assertEqual(reward["base_reward"], CORRECT_FLAG_REWARD)
        self.assertEqual(reward["logic_state"], "probabilistic")
        self.assertEqual(reward["logic_match"], "probabilistic_flag")
        self.assertEqual(reward["logic_bonus"], PROBABILISTIC_FLAG_PENALTY)
        self.assertTrue(game.get_tile(0, 0).is_flagged)

    def test_wrong_flag_returns_penalty(self) -> None:
        game = build_game_from_layout(width=3, height=3, mines={(0, 0)})

        reward = calculate_reward_components(game, {"action": "flag", "x": 2, "y": 2})

        self.assertEqual(reward["base_reward"], WRONG_FLAG_PENALTY)
        self.assertEqual(reward["logic_state"], "probabilistic")
        self.assertEqual(reward["logic_bonus"], PROBABILISTIC_FLAG_PENALTY)
        self.assertTrue(game.get_tile(2, 2).is_flagged)

    def test_forced_safe_reveal_gets_logic_and_frontier_bonus(self) -> None:
        game = build_game_from_layout(
            width=3,
            height=2,
            mines={(0, 0)},
            revealed={(1, 0), (0, 1)},
            flagged={(0, 0)},
        )

        reward = apply_move_and_get_reward(game, {"action": "reveal", "x": 1, "y": 1})

        self.assertEqual(reward, NUMBER_REVEAL_REWARD + DETERMINISTIC_MATCH_BONUS + FRONTIER_PROGRESS_BONUS)

    def test_forced_flag_gets_logic_frontier_and_win_bonus(self) -> None:
        game = build_game_from_layout(
            width=2,
            height=2,
            mines={(0, 0)},
            revealed={(1, 0), (0, 1), (1, 1)},
        )

        reward = apply_move_and_get_reward(game, {"action": "flag", "x": 0, "y": 0})

        self.assertEqual(reward, CORRECT_FLAG_REWARD + DETERMINISTIC_MATCH_BONUS + FRONTIER_PROGRESS_BONUS + WIN_REWARD)
        self.assertIs(game.status, GameStatus.WON)

    def test_guessing_elsewhere_when_forced_move_exists_gets_missed_deterministic_penalty(self) -> None:
        game = build_game_from_layout(
            width=4,
            height=3,
            mines={(0, 0), (3, 2)},
            revealed={(1, 0), (0, 1)},
            flagged={(0, 0)},
        )

        reward = apply_move_and_get_reward(game, {"action": "flag", "x": 3, "y": 0})

        self.assertEqual(reward, WRONG_FLAG_PENALTY + MISSED_DETERMINISTIC_PENALTY)

    def test_flagging_forced_safe_tile_gets_missed_deterministic_penalty_and_frontier_bonus(self) -> None:
        game = build_game_from_layout(
            width=3,
            height=2,
            mines={(0, 0)},
            revealed={(1, 0), (0, 1)},
            flagged={(0, 0)},
        )

        reward = apply_move_and_get_reward(game, {"action": "flag", "x": 1, "y": 1})

        self.assertEqual(reward, WRONG_FLAG_PENALTY + MISSED_DETERMINISTIC_PENALTY + FRONTIER_PROGRESS_BONUS)

    def test_redundant_reveal_returns_custom_penalty(self) -> None:
        game = build_game_from_layout(width=3, height=3, mines={(0, 0)}, revealed={(1, 0)})

        reward = apply_move_and_get_reward(game, {"action": "reveal", "x": 1, "y": 0})

        self.assertEqual(reward, INVALID_REVEAL_PENALTY)
        self.assertTrue(game.get_tile(1, 0).is_revealed)

    def test_flagging_revealed_tile_returns_custom_penalty(self) -> None:
        game = build_game_from_layout(width=3, height=3, mines={(0, 0)}, revealed={(1, 0)})

        reward = apply_move_and_get_reward(game, {"action": "flag", "x": 1, "y": 0})

        self.assertEqual(reward, INVALID_FLAG_PENALTY)
        self.assertFalse(game.get_tile(1, 0).is_flagged)

    def test_flagging_already_flagged_tile_returns_custom_penalty(self) -> None:
        game = build_game_from_layout(width=3, height=3, mines={(0, 0)}, flagged={(2, 2)})

        reward = apply_move_and_get_reward(game, {"action": "flag", "x": 2, "y": 2})

        self.assertEqual(reward, INVALID_REPEAT_FLAG_PENALTY)
        self.assertTrue(game.get_tile(2, 2).is_flagged)

    def test_reward_components_are_separated(self) -> None:
        game = build_game_from_layout(
            width=3,
            height=2,
            mines={(0, 0)},
            revealed={(1, 0), (0, 1)},
            flagged={(0, 0)},
        )

        reward = calculate_reward_components(game, {"action": "reveal", "x": 1, "y": 1})

        self.assertEqual(
            reward,
            {
                "base_reward": NUMBER_REVEAL_REWARD,
                "logic_bonus": DETERMINISTIC_MATCH_BONUS,
                "logic_state": "deterministic",
                "logic_match": "matched",
                "mine_probability": 0.0,
                "candidate_rank": None,
                "deterministic_candidate_count": 3,
                "progress_bonus": FRONTIER_PROGRESS_BONUS,
                "terminal_bonus": 0.0,
                "unclipped_total_reward": NUMBER_REVEAL_REWARD + DETERMINISTIC_MATCH_BONUS + FRONTIER_PROGRESS_BONUS,
                "total_reward": NUMBER_REVEAL_REWARD + DETERMINISTIC_MATCH_BONUS + FRONTIER_PROGRESS_BONUS,
            },
        )

    def test_deterministic_move_scores_higher_than_legal_non_deterministic_choice(self) -> None:
        forced_game = build_game_from_layout(
            width=3,
            height=2,
            mines={(0, 0)},
            revealed={(1, 0), (0, 1)},
            flagged={(0, 0)},
        )
        non_forced_game = build_game_from_layout(
            width=3,
            height=2,
            mines={(0, 0)},
            revealed={(1, 0), (0, 1)},
            flagged={(0, 0)},
        )

        forced_reward = apply_move_and_get_reward(forced_game, {"action": "reveal", "x": 1, "y": 1})
        non_forced_reward = apply_move_and_get_reward(non_forced_game, {"action": "flag", "x": 2, "y": 1})

        self.assertGreater(forced_reward, non_forced_reward)

    @patch("minesweeper.rl_rewards.ProbabilisticSolver.calculate_probabilities")
    @patch("minesweeper.rl_rewards.ranked_solver_actions")
    def test_lower_risk_ranked_guess_scores_higher_than_higher_risk_guess(
        self,
        ranked_solver_actions_mock,
        calculate_probabilities_mock,
    ) -> None:
        game_low = build_game_from_layout(width=3, height=3, mines={(0, 0)})
        game_high = build_game_from_layout(width=3, height=3, mines={(0, 0)})

        ranked_solver_actions_mock.return_value = []
        calculate_probabilities_mock.side_effect = [
            {(0, 1): 0.10, (2, 2): 0.70},
            {(0, 1): 0.10, (2, 2): 0.70},
        ]

        low_reward = apply_move_and_get_reward(game_low, {"action": "reveal", "x": 1, "y": 0})
        high_reward = apply_move_and_get_reward(game_high, {"action": "reveal", "x": 2, "y": 2})

        self.assertGreater(low_reward, high_reward)

    @patch("minesweeper.rl_rewards.ProbabilisticSolver.calculate_probabilities")
    @patch("minesweeper.rl_rewards.ranked_solver_actions")
    def test_ranked_guesses_receive_rank_bonus(
        self,
        ranked_solver_actions_mock,
        calculate_probabilities_mock,
    ) -> None:
        from minesweeper.solve_algo import CandidateAction

        game_rank_1 = build_game_from_layout(width=3, height=3, mines={(0, 0)})
        game_rank_2 = build_game_from_layout(width=3, height=3, mines={(0, 0)})

        ranked_solver_actions_mock.return_value = [
            CandidateAction(action="reveal", row=0, col=1, move_type="probabilistic", mine_probability=0.10),
            CandidateAction(action="reveal", row=1, col=0, move_type="probabilistic", mine_probability=0.10),
        ]
        calculate_probabilities_mock.side_effect = [
            {(0, 1): 0.10, (1, 0): 0.10},
            {(0, 1): 0.10, (1, 0): 0.10},
        ]

        rank_1_reward = apply_move_and_get_reward(game_rank_1, {"action": "reveal", "x": 1, "y": 0})
        rank_2_reward = apply_move_and_get_reward(game_rank_2, {"action": "reveal", "x": 0, "y": 1})

        self.assertGreater(rank_1_reward, rank_2_reward)

    @patch("minesweeper.rl_rewards.ProbabilisticSolver.calculate_probabilities")
    @patch("minesweeper.rl_rewards.ranked_solver_actions")
    def test_probabilistic_flag_gets_logic_penalty(
        self,
        ranked_solver_actions_mock,
        calculate_probabilities_mock,
    ) -> None:
        game = build_game_from_layout(width=3, height=3, mines={(0, 0)})

        ranked_solver_actions_mock.return_value = []
        calculate_probabilities_mock.return_value = {(1, 1): 0.40}

        reward = calculate_reward_components(game, {"action": "flag", "x": 1, "y": 1})

        self.assertEqual(reward["logic_state"], "probabilistic")
        self.assertEqual(reward["logic_match"], "probabilistic_flag")
        self.assertEqual(reward["logic_bonus"], PROBABILISTIC_FLAG_PENALTY)


if __name__ == "__main__":
    unittest.main()
