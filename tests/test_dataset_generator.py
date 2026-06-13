from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_generator import convert_file, transition_to_chat_example


SAMPLE_RECORD = {
    "compact_board_before": [
        [".", ".", "."],
        [".", "1", "."],
        [".", ".", "."],
    ],
    "compact_board_after": [
        [".", ".", "."],
        [".", "1", "."],
        [".", "1", "."],
    ],
    "action": {"action": "reveal", "x": 1, "y": 2},
    "score": 2,
    "metadata": {
        "game_id": 7,
        "step": 3,
        "current_state": "in_progress",
        "move_type": "deterministic",
        "mine_probability_at_action": 0.0,
        "board_size": [3, 3],
        "mine_count": 1,
        "output_format": "compact",
    },
}


class DatasetGeneratorTests(unittest.TestCase):
    def test_transition_to_chat_example_uses_board_and_action(self) -> None:
        example = transition_to_chat_example(SAMPLE_RECORD, system_prompt="system prompt")

        self.assertEqual(example["messages"][0]["role"], "system")
        self.assertEqual(example["messages"][1]["role"], "user")
        self.assertEqual(example["messages"][2]["role"], "assistant")
        self.assertIn(". 1 .", example["messages"][1]["content"])
        self.assertEqual(
            json.loads(example["messages"][2]["content"]),
            {"action": "reveal", "x": 1, "y": 2},
        )

    def test_convert_file_writes_valid_chat_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "raw.jsonl"
            output_path = tmp_path / "nested" / "finetune.jsonl"
            system_prompt_path = tmp_path / "system_prompt.md"

            input_path.write_text(json.dumps(SAMPLE_RECORD) + "\n", encoding="utf-8")
            system_prompt_path.write_text("system prompt", encoding="utf-8")

            summary = convert_file(input_path, output_path, system_prompt_path)

            self.assertEqual(summary.input_records, 1)
            self.assertEqual(summary.output_examples, 1)
            self.assertEqual(summary.skipped_records, 0)

            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["messages"][0]["content"], "system prompt")

    def test_convert_file_can_skip_terminal_losses(self) -> None:
        losing_record = dict(SAMPLE_RECORD)
        losing_record["metadata"] = dict(SAMPLE_RECORD["metadata"])
        losing_record["metadata"]["current_state"] = "lost"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "raw.jsonl"
            output_path = tmp_path / "finetune.jsonl"

            input_path.write_text(json.dumps(losing_record) + "\n", encoding="utf-8")

            summary = convert_file(input_path, output_path, skip_terminal_losses=True)

            self.assertEqual(summary.input_records, 1)
            self.assertEqual(summary.output_examples, 0)
            self.assertEqual(summary.skipped_records, 1)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
