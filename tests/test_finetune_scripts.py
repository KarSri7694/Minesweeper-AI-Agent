from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_action_distribution import analyze_examples
from inspect_finetune_data import inspect_examples
from split_finetune_data import split_file, split_records


def make_example(action: str = "reveal", x: int = 1, y: int = 2) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {
                "role": "user",
                "content": (
                    "Current Minesweeper board state:\n"
                    "Board size: 3 rows x 3 columns\n"
                    "Mine count: 1\n"
                    "Current step: 1\n\n"
                    "Board:\n"
                    ". . .\n"
                    ". 1 .\n"
                    ". . .\n\n"
                    "Choose the next best move."
                ),
            },
            {"role": "assistant", "content": json.dumps({"action": action, "x": x, "y": y})},
        ]
    }


class FinetuneScriptTests(unittest.TestCase):
    def test_split_records_is_deterministic(self) -> None:
        records = [make_example(x=index % 3, y=index // 3) for index in range(10)]

        train_a, val_a = split_records(records, val_ratio=0.2, seed=42)
        train_b, val_b = split_records(records, val_ratio=0.2, seed=42)

        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(train_a), 8)
        self.assertEqual(len(val_a), 2)

    def test_split_file_validates_input(self) -> None:
        bad_example = make_example()
        bad_example["messages"][2]["content"] = "not json"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.jsonl"
            train_path = tmp_path / "out" / "train.jsonl"
            val_path = tmp_path / "out" / "val.jsonl"
            input_path.write_text(json.dumps(bad_example) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "failed validation"):
                split_file(input_path, train_path, val_path, val_ratio=0.5, seed=1)

    def test_inspect_examples_renders_board_and_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            input_path.write_text(json.dumps(make_example()) + "\n", encoding="utf-8")

            rendered = inspect_examples(input_path, count=1)

            self.assertEqual(len(rendered), 1)
            self.assertIn("System preview: system prompt", rendered[0])
            self.assertIn(". 1 .", rendered[0])
            self.assertIn("'action': 'reveal'", rendered[0])

    def test_analyze_examples_counts_distribution_and_invalid_json(self) -> None:
        invalid = make_example()
        invalid["messages"][2]["content"] = "{bad json"

        stats = analyze_examples(
            [
                make_example("reveal", 0, 0),
                make_example("flag", 1, 2),
                invalid,
            ]
        )

        self.assertEqual(stats["actions"]["reveal"], 1)
        self.assertEqual(stats["actions"]["flag"], 1)
        self.assertEqual(stats["board_sizes"][(3, 3)], 3)
        self.assertEqual(stats["unique_coordinates"], 2)
        self.assertEqual(stats["x_counts"][1], 1)
        self.assertEqual(stats["y_counts"][2], 1)
        self.assertEqual(stats["invalid_assistant_json"], 1)


if __name__ == "__main__":
    unittest.main()
