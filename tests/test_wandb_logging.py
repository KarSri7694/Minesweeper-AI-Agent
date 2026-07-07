from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minesweeper.wandb_logging import summarize_reward_logs


class WandbLoggingTests(unittest.TestCase):
    def test_summarize_reward_logs_aggregates_components(self) -> None:
        reward_logs = [
            {
                "reward": 9.0,
                "component": "move_reward",
                "reward_components": {
                    "base_reward": 5.0,
                    "logic_bonus": 3.0,
                    "logic_state": "deterministic",
                    "logic_match": "matched",
                    "mine_probability": 0.0,
                    "candidate_rank": 1,
                    "deterministic_candidate_count": 2,
                    "progress_bonus": 1.0,
                    "terminal_bonus": 0.0,
                    "unclipped_total_reward": 9.0,
                    "total_reward": 9.0,
                },
            },
            {
                "reward": -6.0,
                "component": "execution_error",
            },
        ]

        metrics = summarize_reward_logs(reward_logs, prefix="reward")

        self.assertEqual(metrics["reward/samples"], 2.0)
        self.assertEqual(metrics["reward/reward_mean"], 1.5)
        self.assertEqual(metrics["reward/reward_min"], -6.0)
        self.assertEqual(metrics["reward/reward_max"], 9.0)
        self.assertEqual(metrics["reward/component/move_reward_count"], 1.0)
        self.assertEqual(metrics["reward/component/execution_error_count"], 1.0)
        self.assertEqual(metrics["reward/base_reward_mean"], 5.0)
        self.assertEqual(metrics["reward/logic_bonus_mean"], 3.0)
        self.assertEqual(metrics["reward/mine_probability_mean"], 0.0)
        self.assertEqual(metrics["reward/candidate_rank_mean"], 1.0)
        self.assertEqual(metrics["reward/deterministic_candidate_count_mean"], 2.0)
        self.assertEqual(metrics["reward/total_reward_mean"], 9.0)

    def test_summarize_reward_logs_returns_empty_for_no_logs(self) -> None:
        self.assertEqual(summarize_reward_logs([]), {})


if __name__ == "__main__":
    unittest.main()
