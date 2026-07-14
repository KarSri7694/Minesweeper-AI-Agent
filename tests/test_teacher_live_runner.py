from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minesweeper.training_pipeline import live_teacher_row_to_sft_row
from utils.add_thinking_from_teacher import (
    build_live_teacher_row,
    is_queue_rate_limit_error,
    is_repeated_reveal_result,
    parse_completion_payload,
    play_live_teacher_sessions,
    render_compact_board,
    should_persist_training_row,
)


class TeacherLiveRunnerTests(unittest.TestCase):
    def test_render_compact_board_formats_grid(self) -> None:
        board = [[".", "1"], ["F", "_"]]
        self.assertEqual(render_compact_board(board), ". 1\nF _")

    def test_build_live_teacher_row_contains_training_fields(self) -> None:
        state_before = {
            "game_id": "game-1",
            "board": [[".", "1"], ["F", "_"]],
            "score": 3,
            "status": "in_progress",
            "flagged_count": 1,
            "mine_count": 2,
            "height": 2,
            "width": 2,
        }
        state_after = {
            "board": [[".", "1"], ["F", "1"]],
            "score": 5,
            "status": "in_progress",
            "flagged_count": 1,
            "move_count": 2,
            "last_move": {
                "action": "reveal",
                "x": 1,
                "y": 1,
                "score_delta": 2,
                "message": "Reveal processed.",
                "changed_tiles": [(1, 1)],
            },
        }

        row = build_live_teacher_row(
            session_id="session-1",
            turn_index=2,
            provider="cerebras",
            model="teacher-x",
            state_before=state_before,
            state_after=state_after,
            system_prompt="system",
            user_prompt="user",
            thinking_text="reasoning",
            response_text='{"action":"reveal","x":1,"y":1}',
            move={"action": "reveal", "x": 1, "y": 1},
            error=None,
        )

        self.assertEqual(row["input"], str(state_before["board"]))
        self.assertEqual(row["output"], '{"action":"reveal","x":1,"y":1}')
        self.assertEqual(row["thinking_text"], "reasoning")
        self.assertEqual(row["score_after"], 5)
        self.assertEqual(row["status_after"], "in_progress")

    def test_live_teacher_row_converts_to_sft_row(self) -> None:
        live_row = {
            "session_id": "session-1",
            "game_id": "game-1",
            "turn_index": 1,
            "input": "[['.', '1'], ['F', '_']]",
            "thinking_text": "reasoning",
            "output": '{"action":"flag","x":0,"y":0}',
            "max_mines": 2,
            "max_rows": 2,
            "max_columns": 2,
            "snapshot": "{'example': 1}",
        }

        sft_row = live_teacher_row_to_sft_row(live_row)

        self.assertEqual(sft_row["input"], live_row["input"])
        self.assertEqual(sft_row["output"], live_row["output"])
        self.assertEqual(sft_row["thinking"], "reasoning")

    def test_parse_completion_payload_accepts_thinking_completion(self) -> None:
        payload = parse_completion_payload(
            "<think>\nlook at constraints\n</think>\n\n{\"action\":\"flag\",\"x\":0,\"y\":0}",
            prompt_mode="thinking",
        )

        self.assertEqual(payload, {"action": "flag", "x": 0, "y": 0})

    def test_jsonl_round_trip_for_turn_row(self) -> None:
        row = {
            "session_id": "session-1",
            "game_id": "game-1",
            "turn_index": 1,
            "input": "[['.', '.']]",
            "output": '{"action":"reveal","x":0,"y":0}',
            "max_mines": 1,
            "max_rows": 1,
            "max_columns": 2,
            "thinking_text": "reasoning",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(loaded["output"], row["output"])
        self.assertEqual(loaded["thinking_text"], row["thinking_text"])

    def test_should_persist_training_row_rejects_error_rows(self) -> None:
        self.assertFalse(
            should_persist_training_row(
                move={"action": "reveal", "x": 0, "y": 0},
                state_after={"status": "in_progress", "last_move": {"action": "reveal"}},
                error="json decode error",
            )
        )

    def test_should_persist_training_row_rejects_mine_reveals(self) -> None:
        self.assertFalse(
            should_persist_training_row(
                move={"action": "reveal", "x": 0, "y": 0},
                state_after={
                    "status": "lost",
                    "end_reason": "revealed_mine",
                    "last_move": {"action": "reveal"},
                },
                error=None,
            )
        )

    def test_should_persist_training_row_accepts_clean_move_rows(self) -> None:
        self.assertTrue(
            should_persist_training_row(
                move={"action": "flag", "x": 1, "y": 1},
                state_after={
                    "status": "in_progress",
                    "end_reason": None,
                    "last_move": {"action": "flag"},
                },
                error=None,
            )
        )

    def test_is_queue_rate_limit_error_detects_provider_queue_limit(self) -> None:
        error = Exception(
            "Error code: 429 - {'type': 'too_many_requests_error', 'code': 'queue_exceeded'}"
        )

        self.assertTrue(is_queue_rate_limit_error(error))
        self.assertFalse(is_queue_rate_limit_error(ValueError("json decode error")))

    def test_repeated_reveal_result_is_retryable_and_not_persisted(self) -> None:
        move = {"action": "reveal", "x": 2, "y": 1}
        state_after = {
            "status": "in_progress",
            "last_move": {
                "action": "reveal",
                "message": "Tile already revealed.",
            },
        }

        self.assertTrue(is_repeated_reveal_result(move=move, state_after=state_after))
        self.assertFalse(
            should_persist_training_row(
                move=move,
                state_after=state_after,
                error=None,
            )
        )

    def test_play_live_teacher_sessions_rejects_non_positive_count(self) -> None:
        with self.assertRaises(ValueError):
            play_live_teacher_sessions(
                num_games=0,
                api_base_url="http://127.0.0.1:8000",
                provider="cerebras",
                model="teacher-x",
                api_key="secret",
                base_url=None,
                output_path=Path("dataset/live_teacher_thinking.jsonl"),
                transcript_path=Path("dataset/live_teacher_session.log"),
                width=5,
                height=5,
                mine_density=0.15,
                max_turns=10,
                max_tokens=100,
                turn_delay=0.0,
                rate_limit_cooldown=0.0,
                launch_gui=False,
            )


if __name__ == "__main__":
    unittest.main()
