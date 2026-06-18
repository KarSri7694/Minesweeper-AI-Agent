from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_parser import connect_to_llm, create_game, deterministic_move, get_game_state, get_ollama_num_predict, get_ollama_base_url, make_move, parse_llm_response, probability_candidates, probability_move, use_llm_from_start, validate_move
from minesweeper.engine import GameEngine


class LlmParserTests(unittest.TestCase):
    def test_use_llm_from_start_defaults_off_and_can_be_enabled(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            self.assertFalse(use_llm_from_start())

        with patch.dict("os.environ", {"LLM_FROM_START": "1"}, clear=False):
            self.assertTrue(use_llm_from_start())

    def test_get_ollama_num_predict_defaults_and_can_be_overridden(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            self.assertEqual(get_ollama_num_predict(), 20)

        with patch.dict("os.environ", {"OLLAMA_NUM_PREDICT": "12"}, clear=False):
            self.assertEqual(get_ollama_num_predict(), 12)

    def test_connect_to_llm_prefers_env_vars_for_remote_host_and_model(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OLLAMA_HOST": "https://preferred-remote-ollama.test/",
                "OLLAMA_URL": "https://example-remote-ollama.test/",
                "OLLAMA_MODEL": "finetuned-model",
                "OLLAMA_TIMEOUT_SECONDS": "123",
            },
            clear=False,
        ):
            llm = connect_to_llm()

        self.assertEqual(llm["base_url"], "https://preferred-remote-ollama.test")
        self.assertEqual(llm["model"], "finetuned-model")
        self.assertEqual(llm["timeout"], 123)

    def test_get_ollama_base_url_prefers_host_first(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OLLAMA_HOST": "https://host-first.test/",
                "OLLAMA_BASE_URL": "https://base-url-second.test/",
                "OLLAMA_URL": "https://url-third.test/",
            },
            clear=False,
        ):
            self.assertEqual(get_ollama_base_url(), "https://host-first.test")

    def test_direct_backend_creates_and_moves_without_api(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "GAME_BACKEND": "direct",
                "GAME_WIDTH": "4",
                "GAME_HEIGHT": "4",
                "GAME_MINE_DENSITY": "0.15",
                "GAME_SEED": "3",
            },
        ):
            game = create_game()

        self.assertIsInstance(game, GameEngine)
        move_response = make_move(game, "reveal", 2, 2)
        state = get_game_state(game)

        self.assertIsNotNone(move_response)
        self.assertEqual(state["width"], 4)
        self.assertGreaterEqual(state["score"], 1)

    def test_deterministic_move_flags_forced_mine(self) -> None:
        game_state = {
            "board": [
                ["1", "."],
                ["_", "_"],
            ],
            "move_count": 1,
        }

        self.assertEqual(deterministic_move(game_state), {"action": "flag", "x": 1, "y": 0})

    def test_deterministic_move_reveals_when_number_is_satisfied(self) -> None:
        game_state = {
            "board": [
                ["1", "F"],
                [".", "_"],
            ],
            "move_count": 1,
        }

        self.assertEqual(deterministic_move(game_state), {"action": "reveal", "x": 0, "y": 1})

    def test_parse_llm_response_extracts_json_from_explanation(self) -> None:
        response = """
        I will reveal a low-risk tile.

        ```json
        {"action": "reveal", "row": 4, "col": 3}
        ```
        """

        self.assertEqual(parse_llm_response(response), ("reveal", 4, 3))

    def test_parse_llm_response_accepts_python_dict_style_move(self) -> None:
        response = "{'action': 'flag', 'row': 15, 'col': 2}"

        self.assertEqual(parse_llm_response(response), ("flag", 15, 2))

    def test_probability_move_prefers_lower_risk_tile(self) -> None:
        game_state = {
            "board": [
                ["1", ".", "."],
                ["1", ".", "."],
                [".", "_", "_"],
            ],
            "mine_count": 2,
            "flagged_count": 0,
            "move_count": 1,
        }

        self.assertEqual(probability_move(game_state), {"action": "reveal", "x": 0, "y": 2})

    def test_probability_candidates_surface_ties(self) -> None:
        game_state = {
            "board": [
                [".", ".", "."],
                [".", "1", "."],
                [".", ".", "."],
            ],
            "mine_count": 1,
            "flagged_count": 0,
            "move_count": 1,
        }

        candidates = probability_candidates(game_state)
        self.assertEqual(len(candidates), 8)
        self.assertTrue(all(abs(probability - 0.125) < 1e-9 for _, probability in candidates))

    def test_validate_move_rejects_revealed_tile(self) -> None:
        game_state = {
            "board": [
                ["1", "."],
                ["_", "_"],
            ],
        }

        self.assertIsNone(validate_move({"action": "reveal", "x": 0, "y": 0}, game_state))
        self.assertEqual(
            validate_move({"action": "flag", "row": 0, "col": 1}, game_state),
            {"action": "flag", "x": 1, "y": 0, "row": 0, "col": 1},
        )

    def test_validate_move_does_not_swap_row_and_col(self) -> None:
        game_state = {
            "board": [
                ["1", "1", "1", "1"],
                [".", "1", ".", "."],
                [".", ".", ".", "."],
                [".", ".", ".", "."],
            ],
        }

        self.assertIsNone(validate_move({"action": "reveal", "row": 0, "col": 3}, game_state))
        self.assertEqual(
            validate_move({"action": "reveal", "row": 1, "col": 0}, game_state),
            {"action": "reveal", "x": 0, "y": 1, "row": 1, "col": 0},
        )


if __name__ == "__main__":
    unittest.main()
