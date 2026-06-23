from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_parser import (
    LLMResponseError,
    MoveValidationError,
    build_user_prompt,
    parse_llm_response,
    play_game,
    validate_move,
)


def state(board: list[list[str]], **overrides: object) -> dict:
    game_state = {
        "board": board,
        "width": len(board[0]),
        "height": len(board),
        "mine_count": 2,
        "flagged_count": 0,
        "score": 0,
        "move_count": 0,
        "status": "in_progress",
    }
    game_state.update(overrides)
    return game_state


class LLMParserTests(unittest.TestCase):
    def test_parser_accepts_valid_json(self) -> None:
        self.assertEqual(parse_llm_response('{"action":"reveal","x":2,"y":3}'), ("reveal", 2, 3))

    def test_parser_accepts_markdown_fenced_json(self) -> None:
        response = "```json\n{\n  \"action\": \"flag\", \"x\": 1, \"y\": 0\n}\n```"
        self.assertEqual(parse_llm_response(response), ("flag", 1, 0))

    def test_parser_rejects_python_dict_syntax(self) -> None:
        with self.assertRaises(LLMResponseError):
            parse_llm_response("{'action': 'reveal', 'x': 0, 'y': 0}")

    def test_validator_rejects_revealed_and_out_of_bounds_cells(self) -> None:
        game_state = state([["0", "."], [".", "F"]])
        with self.assertRaisesRegex(MoveValidationError, "not a hidden"):
            validate_move(game_state, "reveal", 0, 0)
        with self.assertRaisesRegex(MoveValidationError, "outside"):
            validate_move(game_state, "flag", 2, 0)

    def test_validator_accepts_hidden_cell_for_both_actions(self) -> None:
        game_state = state([[".", "0"], [".", "F"]])
        validate_move(game_state, "reveal", 0, 0)
        validate_move(game_state, "flag", 0, 1)

    def test_prompt_starts_with_fine_tuning_format(self) -> None:
        prompt = build_user_prompt(state([["."]]))
        self.assertTrue(prompt.startswith("Board State: [['.']]\nMax Mines: 2\nMax Rows: 1\nMax Columns: 1"))
        self.assertIn("Status: in_progress", prompt)

    def test_repeated_invalid_move_stops_after_retry_limit_without_api_call(self) -> None:
        # (0, 0) has already been revealed. Every completion repeats it.
        game_state = state([["0", "."], [".", "."]])

        async def fake_response(*_args: object, **_kwargs: object) -> object:
            return object()

        async def fake_consume(_response: object) -> str:
            return '{"action":"reveal","x":0,"y":0}'

        with (
            patch("llm_parser.get_game_state", return_value=game_state),
            patch("llm_parser.get_llm_response", new=AsyncMock(side_effect=fake_response)),
            patch("llm_parser._consume_stream", new=AsyncMock(side_effect=fake_consume)),
            patch("llm_parser.make_move") as make_move,
        ):
            final_state = asyncio.run(play_game("test-game", object(), max_consecutive_failures=3))

        self.assertEqual(final_state, game_state)
        self.assertEqual(make_move.call_count, 0)


if __name__ == "__main__":
    unittest.main()
