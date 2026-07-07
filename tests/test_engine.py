from __future__ import annotations

import random
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minesweeper.engine import GameConfig, GameEngine, GameStatus


def build_game_from_compact_layout(
    layout: list[list[str]],
    *,
    rng_seed: int = 0,
    lost: bool = False,
) -> GameEngine:
    height = len(layout)
    width = len(layout[0])
    mine_count = sum(cell == "B" for row in layout for cell in row)
    game = GameEngine(
        config=GameConfig(width=width, height=height, mine_density=mine_count / (width * height)),
        rng=random.Random(rng_seed),
    )
    snapshot = game.snapshot()
    snapshot["mines_placed"] = True
    snapshot["rng_state"] = random.Random(rng_seed).getstate()
    snapshot["status"] = GameStatus.LOST.value if lost else GameStatus.IN_PROGRESS.value
    snapshot["end_reason"] = "revealed_mine" if lost else None

    mines: set[tuple[int, int]] = set()
    for y, row in enumerate(layout):
        for x, cell in enumerate(row):
            tile = snapshot["board"][y][x]
            tile["is_revealed"] = cell.isdigit()
            tile["is_flagged"] = cell == "F"
            tile["flag_score_applied"] = cell == "F"
            tile["is_mine"] = cell == "B"
            if cell == "B":
                mines.add((x, y))

    for y, row in enumerate(layout):
        for x, _cell in enumerate(row):
            tile = snapshot["board"][y][x]
            if tile["is_mine"]:
                tile["adjacent_mines"] = 0
                continue
            adjacent = 0
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if (nx, ny) != (x, y) and (nx, ny) in mines:
                        adjacent += 1
            tile["adjacent_mines"] = adjacent

    return GameEngine.from_snapshot(snapshot)


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

        self.assertEqual(first_flag.score_delta, game.score_rules.wrong_flag)
        self.assertEqual(unflag.score_delta, 0)
        self.assertEqual(second_flag.score_delta, 0)
        self.assertGreaterEqual(game.score, game.score_rules.wrong_flag)

    def test_win_requires_safe_reveals_and_all_mines_flagged(self) -> None:
        game = GameEngine(config=GameConfig(width=2, height=2, mine_density=0.25))
        game.reveal(0, 0)

        mine_tile = next(tile for tile in game.iter_tiles() if tile.is_mine)
        for tile in game.iter_tiles():
            if not tile.is_mine and not tile.is_revealed:
                game.reveal(tile.x, tile.y)

        self.assertIs(game.status, GameStatus.IN_PROGRESS)

        flag_result = game.flag(mine_tile.x, mine_tile.y)
        self.assertEqual(flag_result.score_delta, game.score_rules.correct_flag)
        self.assertIs(game.status, GameStatus.WON)
        self.assertGreaterEqual(game.score, game.score_rules.win_bonus)

    def test_revealing_same_tile_twice_is_noop_during_in_progress(self) -> None:
        game = GameEngine(config=GameConfig(width=5, height=5))
        first_result = game.reveal(0, 0)
        snapshot_before = game.snapshot()

        second_result = game.reveal(0, 0)

        self.assertGreaterEqual(first_result.score_delta, 1)
        self.assertEqual(second_result.score_delta, 0)
        self.assertEqual(second_result.changed_tiles, [])
        self.assertEqual(second_result.message, "Tile already revealed.")
        self.assertEqual(game.snapshot(), snapshot_before)

    def test_revealing_flagged_tile_is_noop(self) -> None:
        game = GameEngine(config=GameConfig(width=5, height=5))
        tile = game.get_tile(1, 1)
        game.flag(tile.x, tile.y)
        snapshot_before = game.snapshot()

        result = game.reveal(tile.x, tile.y)

        self.assertTrue(tile.is_flagged)
        self.assertFalse(tile.is_revealed)
        self.assertEqual(result.score_delta, 0)
        self.assertEqual(result.changed_tiles, [])
        self.assertEqual(result.message, "Tile is flagged.")
        self.assertEqual(game.snapshot(), snapshot_before)

    def test_revealing_revealed_tile_after_win_is_noop(self) -> None:
        game = GameEngine(config=GameConfig(width=2, height=2, mine_density=0.25))
        game.reveal(0, 0)

        mine_tile = next(tile for tile in game.iter_tiles() if tile.is_mine)
        repeated_tile = next(tile for tile in game.iter_tiles() if tile.is_revealed)
        for tile in game.iter_tiles():
            if not tile.is_mine and not tile.is_revealed:
                game.reveal(tile.x, tile.y)
        game.flag(mine_tile.x, mine_tile.y)
        snapshot_before = game.snapshot()

        result = game.reveal(repeated_tile.x, repeated_tile.y)

        self.assertIs(game.status, GameStatus.WON)
        self.assertEqual(result.score_delta, 0)
        self.assertEqual(result.changed_tiles, [])
        self.assertEqual(result.message, "Tile already revealed.")
        self.assertEqual(game.snapshot(), snapshot_before)

    def test_revealing_revealed_tile_after_loss_is_noop(self) -> None:
        layout = [
            ["1", ".", "."],
            ["B", ".", "."],
            [".", ".", "."],
        ]
        game = build_game_from_compact_layout(layout, rng_seed=42, lost=True)
        repeated_tile = game.get_tile(0, 0)
        snapshot_before = game.snapshot()

        result = game.reveal(repeated_tile.x, repeated_tile.y)

        self.assertIs(game.status, GameStatus.LOST)
        self.assertEqual(result.score_delta, 0)
        self.assertEqual(result.changed_tiles, [])
        self.assertEqual(result.message, "Tile already revealed.")
        self.assertEqual(game.snapshot(), snapshot_before)

    def test_revealing_hidden_tile_after_game_end_still_raises(self) -> None:
        won_game = GameEngine(config=GameConfig(width=2, height=2, mine_density=0.25))
        won_snapshot = won_game.snapshot()
        won_snapshot["status"] = GameStatus.WON.value
        won_snapshot["end_reason"] = "all_mines_flagged"
        won_snapshot["mines_placed"] = True
        won_snapshot["board"][0][0]["is_revealed"] = True
        won_snapshot["board"][0][0]["adjacent_mines"] = 1
        won_game = GameEngine.from_snapshot(won_snapshot)
        hidden_lost_game = build_game_from_compact_layout(
            [
                ["1", ".", "."],
                ["B", ".", "."],
                [".", ".", "."],
            ],
            rng_seed=7,
            lost=True,
        )

        with self.assertRaisesRegex(ValueError, "Game is already won."):
            won_game.reveal(1, 0)
        with self.assertRaisesRegex(ValueError, "Game is already lost."):
            hidden_lost_game.reveal(1, 0)

    def test_snapshot_round_trip_restores_exact_same_state(self) -> None:
        game = GameEngine(config=GameConfig(width=5, height=5, mine_density=0.2))
        game.reveal(0, 0)

        hidden_safe_tile = next(
            tile for tile in game.iter_tiles() if not tile.is_mine and not tile.is_revealed
        )
        game.flag(hidden_safe_tile.x, hidden_safe_tile.y)

        snapshot = game.snapshot()
        restored = GameEngine.from_snapshot(snapshot)

        self.assertEqual(restored.snapshot(), snapshot)
        self.assertEqual(restored.visible_state(), game.visible_state())
        self.assertEqual(restored.compact_state(), game.compact_state())

        next_safe_tile = next(
            tile
            for tile in game.iter_tiles()
            if not tile.is_mine and not tile.is_revealed and not tile.is_flagged
        )

        original_result = game.reveal(next_safe_tile.x, next_safe_tile.y)
        restored_result = restored.reveal(next_safe_tile.x, next_safe_tile.y)

        self.assertEqual(restored_result.score_delta, original_result.score_delta)
        self.assertEqual(restored_result.changed_tiles, original_result.changed_tiles)
        self.assertEqual(restored_result.message, original_result.message)
        self.assertEqual(restored.visible_state(), game.visible_state())
        self.assertEqual(restored.compact_state(), game.compact_state())

    def test_full_board_compact_state_reveals_entire_board_without_mutation(self) -> None:
        game = GameEngine(config=GameConfig(width=4, height=4, mine_density=0.25))
        game.reveal(0, 0)
        snapshot_before = game.snapshot()

        full_state = game.full_board_compact_state()

        self.assertEqual(full_state["game_id"], game.game_id)
        self.assertEqual(full_state["width"], 4)
        self.assertEqual(full_state["height"], 4)
        self.assertEqual(len(full_state["board"]), 4)
        self.assertTrue(any(cell == "B" for row in full_state["board"] for cell in row))

        for tile in game.iter_tiles():
            expected_value = "B" if tile.is_mine else ("0" if tile.adjacent_mines == 0 else str(tile.adjacent_mines))
            self.assertEqual(full_state["board"][tile.y][tile.x], expected_value)

        self.assertEqual(game.snapshot(), snapshot_before)

    def test_full_board_compact_state_requires_mines_to_be_placed(self) -> None:
        game = GameEngine(config=GameConfig(width=4, height=4, mine_density=0.25))

        with self.assertRaisesRegex(ValueError, "Full board is unavailable until mines have been placed"):
            game.full_board_compact_state()

    def test_build_game_from_compact_layout_restores_exact_debug_state(self) -> None:
        layout = [
            [".", ".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", "B", "B", ".", ".", ".", ".", "B"],
            [".", ".", ".", ".", ".", ".", "B", ".", "."],
            [".", ".", "B", ".", ".", "B", ".", ".", "."],
            [".", ".", ".", "B", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", "B", ".", ".", ".", "."],
            [".", "B", "B", ".", ".", "B", ".", ".", "."],
            ["1", "2", "2", "1", "1", "1", "2", ".", "."],
            ["0", "0", "0", "0", "0", "0", "1", "B", "."],
        ]

        game = build_game_from_compact_layout(layout, rng_seed=42, lost=True)

        self.assertEqual(game.compact_state()["board"], layout)
        self.assertEqual(game.full_board_compact_state()["board"][8][7], "B")
        self.assertEqual(game.rng.getstate(), random.Random(42).getstate())
        self.assertIs(game.status, GameStatus.LOST)


if __name__ == "__main__":
    unittest.main()
