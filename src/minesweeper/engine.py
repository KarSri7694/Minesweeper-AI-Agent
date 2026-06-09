from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Iterable
from uuid import uuid4


class GameStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    WON = "won"
    LOST = "lost"


@dataclass(frozen=True)
class ScoreRules:
    safe_reveal: int = 1
    correct_flag: int = 2
    wrong_flag: int = -2
    win_bonus: int = 50


@dataclass(frozen=True)
class GameConfig:
    width: int
    height: int
    mine_density: float = 0.15

    def validate(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("Board dimensions must both be at least 2.")
        if self.width > 50 or self.height > 50:
            raise ValueError("Board dimensions must both be at most 50.")
        if not 0 < self.mine_density < 1:
            raise ValueError("Mine density must be between 0 and 1.")
        if self.width * self.height < 2:
            raise ValueError("Board must contain at least two tiles.")

    @property
    def mine_count(self) -> int:
        total_tiles = self.width * self.height
        return min(total_tiles - 1, max(1, round(total_tiles * self.mine_density)))


@dataclass
class Tile:
    x: int
    y: int
    is_mine: bool = False
    adjacent_mines: int = 0
    is_revealed: bool = False
    is_flagged: bool = False
    flag_score_applied: bool = False


@dataclass
class MoveResult:
    score_delta: int
    changed_tiles: list[tuple[int, int]]
    message: str


@dataclass
class GameEngine:
    config: GameConfig
    score_rules: ScoreRules = field(default_factory=ScoreRules)
    rng: random.Random = field(default_factory=random.Random)
    game_id: str = field(default_factory=lambda: str(uuid4()))
    status: GameStatus = GameStatus.IN_PROGRESS
    score: int = 0
    move_count: int = 0
    end_reason: str | None = None
    _board: list[list[Tile]] = field(init=False, repr=False)
    _mines_placed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.config.validate()
        self._board = [
            [Tile(x=x, y=y) for x in range(self.config.width)]
            for y in range(self.config.height)
        ]

    @property
    def mine_count(self) -> int:
        return self.config.mine_count

    @property
    def flagged_count(self) -> int:
        return sum(tile.is_flagged for tile in self.iter_tiles())

    def iter_tiles(self) -> Iterable[Tile]:
        for row in self._board:
            yield from row

    def get_tile(self, x: int, y: int) -> Tile:
        if not self.in_bounds(x, y):
            raise ValueError(f"Tile ({x}, {y}) is out of bounds.")
        return self._board[y][x]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def neighbors(self, x: int, y: int) -> list[Tile]:
        result: list[Tile] = []
        for ny in range(max(0, y - 1), min(self.config.height, y + 2)):
            for nx in range(max(0, x - 1), min(self.config.width, x + 2)):
                if nx == x and ny == y:
                    continue
                result.append(self._board[ny][nx])
        return result

    def reveal(self, x: int, y: int) -> MoveResult:
        self._ensure_playable()
        tile = self.get_tile(x, y)
        if tile.is_flagged:
            raise ValueError("Flagged tiles cannot be revealed until unflagged.")
        if tile.is_revealed:
            return MoveResult(0, [], "Tile already revealed.")
        if not self._mines_placed:
            self._place_mines_safe_from(x, y)

        self.move_count += 1
        if tile.is_mine:
            tile.is_revealed = True
            self.status = GameStatus.LOST
            self.end_reason = "revealed_mine"
            changed_tiles = [(x, y)]
            return MoveResult(0, changed_tiles, "Mine revealed. Game over.")

        changed_tiles: list[tuple[int, int]] = []
        safe_reveals = self._reveal_region(x, y, changed_tiles)
        score_delta = safe_reveals * self.score_rules.safe_reveal
        self.score += score_delta
        self._maybe_finish_game()
        return MoveResult(score_delta, changed_tiles, "Reveal processed.")

    def flag(self, x: int, y: int) -> MoveResult:
        self._ensure_playable()
        tile = self.get_tile(x, y)
        if tile.is_revealed:
            raise ValueError("Revealed tiles cannot be flagged.")

        self.move_count += 1
        tile.is_flagged = not tile.is_flagged
        score_delta = 0
        if tile.is_flagged and not tile.flag_score_applied:
            tile.flag_score_applied = True
            score_delta = (
                self.score_rules.correct_flag if tile.is_mine else self.score_rules.wrong_flag
            )
            self.score += score_delta

        self._maybe_finish_game()
        message = "Flag placed." if tile.is_flagged else "Flag removed."
        return MoveResult(score_delta, [(x, y)], message)

    def _ensure_playable(self) -> None:
        if self.status is not GameStatus.IN_PROGRESS:
            raise ValueError(f"Game is already {self.status.value}.")

    def _place_mines_safe_from(self, safe_x: int, safe_y: int) -> None:
        all_positions = [
            (x, y)
            for y in range(self.config.height)
            for x in range(self.config.width)
            if not (x == safe_x and y == safe_y)
        ]
        mine_positions = self.rng.sample(all_positions, self.mine_count)
        for mine_x, mine_y in mine_positions:
            self._board[mine_y][mine_x].is_mine = True
        for tile in self.iter_tiles():
            tile.adjacent_mines = sum(neighbor.is_mine for neighbor in self.neighbors(tile.x, tile.y))
        self._mines_placed = True

    def _reveal_region(self, start_x: int, start_y: int, changed_tiles: list[tuple[int, int]]) -> int:
        stack = [(start_x, start_y)]
        revealed_count = 0
        while stack:
            x, y = stack.pop()
            tile = self._board[y][x]
            if tile.is_revealed or tile.is_flagged:
                continue
            tile.is_revealed = True
            changed_tiles.append((x, y))
            revealed_count += 1
            if tile.adjacent_mines == 0:
                for neighbor in self.neighbors(x, y):
                    if not neighbor.is_revealed and not neighbor.is_mine:
                        stack.append((neighbor.x, neighbor.y))
        return revealed_count

    def _maybe_finish_game(self) -> None:
        if self.status is not GameStatus.IN_PROGRESS:
            return
        all_safe_revealed = all(tile.is_revealed for tile in self.iter_tiles() if not tile.is_mine)
        all_mines_flagged = all(tile.is_flagged for tile in self.iter_tiles() if tile.is_mine)
        no_false_flags = all(not tile.is_flagged for tile in self.iter_tiles() if not tile.is_mine)
        if all_safe_revealed and all_mines_flagged and no_false_flags:
            self.status = GameStatus.WON
            self.end_reason = "all_mines_flagged"
            self.score += self.score_rules.win_bonus

    def visible_state(self) -> dict:
        board = []
        for row in self._board:
            board_row = []
            for tile in row:
                board_row.append(self._serialize_tile(tile))
            board.append(board_row)
        return {
            "game_id": self.game_id,
            "status": self.status.value,
            "width": self.config.width,
            "height": self.config.height,
            "mine_count": self.mine_count,
            "flagged_count": self.flagged_count,
            "score": self.score,
            "move_count": self.move_count,
            "end_reason": self.end_reason,
            "board": board,
        }

    def compact_state(self) -> dict:
        board = []
        for row in self._board:
            board.append([self._compact_tile_value(tile) for tile in row])
        return {
            "game_id": self.game_id,
            "status": self.status.value,
            "width": self.config.width,
            "height": self.config.height,
            "mine_count": self.mine_count,
            "flagged_count": self.flagged_count,
            "score": self.score,
            "move_count": self.move_count,
            "end_reason": self.end_reason,
            "board": board,
        }

    def _serialize_tile(self, tile: Tile) -> dict:
        if self.status is GameStatus.IN_PROGRESS:
            if tile.is_flagged:
                state = "flagged"
            elif tile.is_revealed:
                state = "revealed"
            else:
                state = "hidden"
            return {
                "x": tile.x,
                "y": tile.y,
                "state": state,
                "adjacent_mines": tile.adjacent_mines if tile.is_revealed else None,
            }

        terminal_state = "revealed" if tile.is_revealed else "hidden"
        if tile.is_flagged:
            terminal_state = "flagged"
        if tile.is_mine and self.status is GameStatus.LOST:
            terminal_state = "mine"
        return {
            "x": tile.x,
            "y": tile.y,
            "state": terminal_state,
            "adjacent_mines": tile.adjacent_mines if tile.is_revealed or not tile.is_mine else None,
            "is_mine": tile.is_mine,
        }

    def _compact_tile_value(self, tile: Tile) -> str:
        if tile.is_flagged:
            return "F"
        if self.status is GameStatus.LOST and tile.is_mine:
            return "B"
        if not tile.is_revealed:
            return "."
        if tile.adjacent_mines == 0:
            return "0"
        return str(tile.adjacent_mines)


class GameManager:
    def __init__(self, score_rules: ScoreRules | None = None) -> None:
        self.score_rules = score_rules or ScoreRules()
        self._games: dict[str, GameEngine] = {}

    def create_game(
        self,
        width: int,
        height: int,
        mine_density: float = 0.15,
        seed: int | None = None,
    ) -> GameEngine:
        rng = random.Random(seed) if seed is not None else random.Random()
        game = GameEngine(
            config=GameConfig(width=width, height=height, mine_density=mine_density),
            score_rules=self.score_rules,
            rng=rng,
        )
        self._games[game.game_id] = game
        return game

    def get_game(self, game_id: str) -> GameEngine:
        try:
            return self._games[game_id]
        except KeyError as exc:
            raise KeyError(f"Unknown game_id: {game_id}") from exc
