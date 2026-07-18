from __future__ import annotations

import json
import random
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minesweeper.engine import GameConfig, GameEngine
from minesweeper.training_pipeline import (
    DIRECT_GRPO_STAGE,
    THINKING_GRPO_STAGE,
    build_programmatic_reasoning_trace,
    build_reasoning_target,
    build_reward_function,
    build_user_prompt_from_state,
    extract_last_json_object,
    parse_completion_payload,
    prepare_sft_dataframe,
    summarize_local_constraints,
    validate_move,
)


class FakeFrame:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def copy(self):
        return FakeFrame([dict(row) for row in self._rows])

    def apply(self, func, axis=1):
        if axis != 1:
            raise AssertionError("These tests only support row-wise apply.")
        return [func(row) for row in self._rows]

    def __setitem__(self, key, values):
        for row, value in zip(self._rows, values):
            row[key] = value

    def __getitem__(self, key):
        return [row[key] for row in self._rows]

    @property
    def rows(self):
        return self._rows


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "input": '[["_", "1", "."], ["_", "1", "F"], ["_", "_", "."]]',
            "output": '{"action":"reveal","x":2,"y":0}',
            "max_mines": 2,
            "max_rows": 3,
            "max_columns": 3,
            "candidate_rank": 1,
            "candidate_count": 2,
            "move_type": "probabilistic",
            "mine_probability_at_action": 0.125,
        }

    def test_build_user_prompt_from_state_formats_board_metadata(self) -> None:
        prompt = build_user_prompt_from_state(
            {
                "board": [[".", "1"], ["F", "_"]],
                "mine_count": 3,
                "height": 2,
                "width": 2,
                "score": 5,
            },
            include_score=True,
        )

        self.assertIn("Board State: [['.', '1'], ['F', '_']]", prompt)
        self.assertIn("x is the column index and y is the row index", prompt)
        self.assertIn("Only choose hidden cells", prompt)
        self.assertIn("Score: 5", prompt)
        self.assertIn("Max Mines: 3", prompt)

    def test_summarize_local_constraints_mentions_neighbor_numbers(self) -> None:
        lines = summarize_local_constraints(self.row["input"], 2, 0)

        self.assertTrue(lines)
        self.assertIn("Cell (1, 0) shows 1", lines[0])

    def test_build_programmatic_reasoning_trace_uses_probabilistic_context(self) -> None:
        trace = build_programmatic_reasoning_trace(self.row)

        self.assertIn("No deterministic move is available", trace)
        self.assertIn("estimated mine probability 0.125", trace)
        self.assertIn('{"action":"reveal","x":2,"y":0}', trace)

    def test_build_reasoning_target_wraps_trace_in_think_tags(self) -> None:
        target = build_reasoning_target(self.row)

        self.assertTrue(target.startswith("<think>\n"))
        self.assertIn("</think>", target)
        self.assertTrue(target.rstrip().endswith('{"action":"reveal","x":2,"y":0}'))

    def test_extract_last_json_object_returns_terminal_payload(self) -> None:
        extracted = extract_last_json_object("<think>\nreason\n</think>\n\n{\"action\":\"flag\",\"x\":1,\"y\":2}")

        self.assertEqual(extracted, '{"action":"flag","x":1,"y":2}')

    def test_parse_completion_payload_accepts_reasoning_then_json(self) -> None:
        payload = parse_completion_payload(
            "<think>\nreason\n</think>\n\n{\"action\":\"flag\",\"x\":1,\"y\":2}",
            prompt_mode="thinking",
        )

        self.assertEqual(payload, {"action": "flag", "x": 1, "y": 2})

    def test_parse_completion_payload_rejects_think_block_in_direct_mode(self) -> None:
        with self.assertRaises(ValueError):
            parse_completion_payload(
                "<think>\nreason\n</think>\n\n{\"action\":\"flag\",\"x\":1,\"y\":2}",
                prompt_mode="direct",
            )

    def test_prepare_sft_dataframe_builds_messages_for_stage(self) -> None:
        frame = FakeFrame([self.row])
        prepared = prepare_sft_dataframe(
            frame,
            stage=THINKING_GRPO_STAGE,
            system_prompt="system",
        )

        messages = prepared.rows[0]["Messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertIn("<think>", messages[2]["content"])

    def test_validate_move_checks_bounds_and_action(self) -> None:
        self.assertTrue(validate_move(3, 3, {"action": "flag", "x": 2, "y": 2}))
        self.assertFalse(validate_move(3, 3, {"action": "boom", "x": 2, "y": 2}))
        self.assertFalse(validate_move(3, 3, {"action": "flag", "x": 4, "y": 2}))

    def test_build_reward_function_penalizes_reasoning_leak_in_direct_mode(self) -> None:
        reward_fn = build_reward_function(DIRECT_GRPO_STAGE)
        game = GameEngine(config=GameConfig(width=2, height=2, mine_density=0.25), rng=random.Random(7))
        game.reveal(0, 0)
        snapshot = game.snapshot()

        rewards = reward_fn(
            completions=['<think>\nreason\n</think>\n\n{"action":"reveal","x":0,"y":0}'],
            rows=[2],
            columns=[2],
            snapshot=[str(snapshot)],
        )

        self.assertEqual(rewards, [-1.0])


if __name__ == "__main__":
    unittest.main()
