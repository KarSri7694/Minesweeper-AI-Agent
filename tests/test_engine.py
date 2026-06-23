from __future__ import annotations

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

        self.assertEqual(first_flag.score_delta, -1)
        self.assertEqual(unflag.score_delta, 0)
        self.assertEqual(second_flag.score_delta, 0)
        self.assertGreaterEqual(game.score, -1)

    def test_win_requires_safe_reveals_and_all_mines_flagged(self) -> None:
        game = GameEngine(config=GameConfig(width=2, height=2, mine_density=0.25))
        game.reveal(0, 0)

        mine_tile = next(tile for tile in game.iter_tiles() if tile.is_mine)
        for tile in game.iter_tiles():
            if not tile.is_mine and not tile.is_revealed:
                game.reveal(tile.x, tile.y)

        self.assertIs(game.status, GameStatus.IN_PROGRESS)

        flag_result = game.flag(mine_tile.x, mine_tile.y)
        self.assertEqual(flag_result.score_delta, 5)
        self.assertIs(game.status, GameStatus.WON)
        self.assertGreaterEqual(game.score, 20)

    def test_revealing_flagged_tile_is_rejected(self) -> None:
        game = GameEngine(config=GameConfig(width=5, height=5))
        tile = game.get_tile(1, 1)
        game.flag(tile.x, tile.y)

        with self.assertRaisesRegex(ValueError, "Flagged tiles cannot be revealed"):
            game.reveal(tile.x, tile.y)

    def test_revealing_an_already_revealed_tile_is_rejected(self) -> None:
        game = GameEngine(config=GameConfig(width=5, height=5))
        game.reveal(0, 0)

        with self.assertRaisesRegex(ValueError, "cannot be revealed again"):
            game.reveal(0, 0)


if __name__ == "__main__":
    unittest.main()
