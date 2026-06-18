from __future__ import annotations

import ast
import json
import random
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minesweeper.engine import GameConfig, GameEngine, GameStatus


class GameEngineTests(unittest.TestCase):
    def test_first_reveal_is_safe(self) -> None:
        game = GameEngine(config=GameConfig(width=5, height=5))
        result = game.reveal(0, 0)
        tile = game.get_tile(0, 0)
        self.assertGreaterEqual(result.score_delta, 1)
        self.assertTrue(tile.is_revealed)
        self.assertFalse(tile.is_mine)
        self.assertIs(game.status, GameStatus.IN_PROGRESS)

    def test_wrong_flag_deducts_once_even_after_toggle(self) -> None:
        game = GameEngine(config=GameConfig(width=5, height=5))
        game.reveal(0, 0)

        safe_tile = next(tile for tile in game.iter_tiles() if not tile.is_mine and not tile.is_revealed)
        first_flag = game.flag(safe_tile.x, safe_tile.y)
        unflag = game.flag(safe_tile.x, safe_tile.y)
        second_flag = game.flag(safe_tile.x, safe_tile.y)

        self.assertEqual(first_flag.score_delta, -2)
        self.assertEqual(unflag.score_delta, 0)
        self.assertEqual(second_flag.score_delta, 0)
        self.assertGreaterEqual(game.score, -2)

    def test_win_requires_safe_reveals_and_all_mines_flagged(self) -> None:
        game = GameEngine(config=GameConfig(width=2, height=2, mine_density=0.25))
        game.reveal(0, 0)

        mine_tile = next(tile for tile in game.iter_tiles() if tile.is_mine)
        for tile in game.iter_tiles():
            if not tile.is_mine and not tile.is_revealed:
                game.reveal(tile.x, tile.y)

        self.assertIs(game.status, GameStatus.IN_PROGRESS)

        flag_result = game.flag(mine_tile.x, mine_tile.y)
        self.assertEqual(flag_result.score_delta, 2)
        self.assertIs(game.status, GameStatus.WON)
        self.assertGreaterEqual(game.score, 50)

    def test_revealing_flagged_tile_is_rejected(self) -> None:
        game = GameEngine(config=GameConfig(width=5, height=5))
        game.reveal(0, 0)
        tile = next(tile for tile in game.iter_tiles() if not tile.is_revealed and not tile.is_mine)
        game.flag(tile.x, tile.y)

        with self.assertRaisesRegex(ValueError, "Flagged tiles cannot be revealed"):
            game.reveal(tile.x, tile.y)

    def test_first_move_must_be_reveal_before_flags_are_allowed(self) -> None:
        game = GameEngine(config=GameConfig(width=5, height=5))

        with self.assertRaisesRegex(ValueError, "First move must be a reveal"):
            game.flag(1, 1)

    def test_compact_state_uses_underscore_for_revealed_zero_tiles(self) -> None:
        game = GameEngine(config=GameConfig(width=5, height=5), rng=random.Random(4))
        game.reveal(0, 0)

        compact_board = game.compact_state()["board"]

        self.assertIn("_", {cell for row in compact_board for cell in row})

    def test_snapshot_round_trip_preserves_mid_game_state(self) -> None:
        game = GameEngine(
            config=GameConfig(width=6, height=6, mine_density=0.2),
            rng=random.Random(7),
        )
        game.reveal(0, 0)
        mine_tile = next(tile for tile in game.iter_tiles() if tile.is_mine and not tile.is_revealed)
        game.flag(mine_tile.x, mine_tile.y)

        restored = GameEngine.from_snapshot(ast.literal_eval(repr(game.snapshot())))

        self.assertEqual(restored.compact_state(), game.compact_state())
        self.assertEqual(restored.score, game.score)
        self.assertEqual(restored.move_count, game.move_count)
        self.assertEqual(restored.status, game.status)

        next_safe_tile = next(
            tile
            for tile in game.iter_tiles()
            if not tile.is_mine and not tile.is_revealed and not tile.is_flagged
        )
        original_result = game.reveal(next_safe_tile.x, next_safe_tile.y)
        restored_result = restored.reveal(next_safe_tile.x, next_safe_tile.y)

        self.assertEqual(restored_result.score_delta, original_result.score_delta)
        self.assertEqual(restored_result.changed_tiles, original_result.changed_tiles)
        self.assertEqual(restored.compact_state(), game.compact_state())

    def test_snapshot_round_trip_preserves_unplaced_mine_rng_state(self) -> None:
        game = GameEngine(
            config=GameConfig(width=5, height=5, mine_density=0.2),
            rng=random.Random(11),
        )
        json_snapshot = json.loads(json.dumps(game.snapshot()))
        restored = GameEngine.from_snapshot(json_snapshot)

        original_result = game.reveal(2, 2)
        restored_result = restored.reveal(2, 2)

        self.assertEqual(restored_result.score_delta, original_result.score_delta)
        self.assertEqual(restored_result.changed_tiles, original_result.changed_tiles)
        self.assertEqual(restored.compact_state(), game.compact_state())


if __name__ == "__main__":
    unittest.main()
