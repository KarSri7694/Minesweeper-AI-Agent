from __future__ import annotations

from .engine import GameEngine

NUMBER_REVEAL_REWARD = 5.0
ZERO_REVEAL_REWARD = 2.0
CORRECT_FLAG_REWARD = 10.0
WRONG_FLAG_PENALTY = -10.0
MINE_REVEAL_PENALTY = -25.0
WIN_REWARD = 50.0
CERTAINTY_BONUS = 3.0
FRONTIER_PROGRESS_BONUS = 1.0
FORCED_MOVE_MISSED_PENALTY = -4.0
CONTRADICTION_PENALTY = -6.0
MIN_FINAL_REWARD = -3.0
MAX_FINAL_REWARD = 3.0


def _find_forced_moves(game: GameEngine) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    forced_safe: set[tuple[int, int]] = set()
    forced_mines: set[tuple[int, int]] = set()

    for tile in game.iter_tiles():
        if not tile.is_revealed or tile.adjacent_mines <= 0:
            continue

        neighbors = game.neighbors(tile.x, tile.y)
        hidden_neighbors = [neighbor for neighbor in neighbors if not neighbor.is_revealed and not neighbor.is_flagged]
        flagged_neighbors = sum(neighbor.is_flagged for neighbor in neighbors)
        remaining_mines = tile.adjacent_mines - flagged_neighbors

        if not hidden_neighbors:
            continue
        if remaining_mines == 0:
            forced_safe.update((neighbor.x, neighbor.y) for neighbor in hidden_neighbors)
        elif remaining_mines == len(hidden_neighbors):
            forced_mines.update((neighbor.x, neighbor.y) for neighbor in hidden_neighbors)

    return forced_safe, forced_mines


def _is_frontier_move(game: GameEngine, x: int, y: int) -> bool:
    return any(neighbor.is_revealed and neighbor.adjacent_mines > 0 for neighbor in game.neighbors(x, y))


def clip_reward(value: float) -> float:
    return max(MIN_FINAL_REWARD, min(MAX_FINAL_REWARD, value))


def calculate_reward_components(game: GameEngine, response: dict) -> dict[str, float]:
    """Return GRPO reward shaping without changing gameplay scoring rules."""
    action = response["action"]
    x = response["x"]
    y = response["y"]
    target_tile = game.get_tile(x, y)

    if target_tile.is_revealed:
        raise ValueError("Revealing an already revealed tile is penalized in GRPO reward shaping.")
    if action == "flag" and target_tile.is_flagged:
        raise ValueError("Flagging an already flagged tile is penalized in GRPO reward shaping.")

    forced_safe, forced_mines = _find_forced_moves(game)
    forced_moves_exist = bool(forced_safe or forced_mines)
    target = (x, y)

    logic_bonus = 0.0
    if action == "reveal":
        if target in forced_safe:
            logic_bonus += CERTAINTY_BONUS
        elif target in forced_mines:
            logic_bonus += CONTRADICTION_PENALTY
        elif forced_moves_exist:
            logic_bonus += FORCED_MOVE_MISSED_PENALTY
    else:
        if target in forced_mines:
            logic_bonus += CERTAINTY_BONUS
        elif target in forced_safe:
            logic_bonus += CONTRADICTION_PENALTY
        elif forced_moves_exist:
            logic_bonus += FORCED_MOVE_MISSED_PENALTY

    progress_bonus = FRONTIER_PROGRESS_BONUS if _is_frontier_move(game, x, y) else 0.0

    if action == "reveal":
        game.reveal(x, y)
        if target_tile.is_mine:
            base_reward = MINE_REVEAL_PENALTY
        elif target_tile.adjacent_mines > 0:
            base_reward = NUMBER_REVEAL_REWARD
        else:
            base_reward = ZERO_REVEAL_REWARD
    else:
        game.flag(x, y)
        if target_tile.is_mine:
            base_reward = CORRECT_FLAG_REWARD
        else:
            base_reward = WRONG_FLAG_PENALTY

    terminal_bonus = WIN_REWARD if game.status.value == "won" else 0.0

    unclipped_total = base_reward + logic_bonus + progress_bonus + terminal_bonus
    return {
        "base_reward": base_reward,
        "logic_bonus": logic_bonus,
        "progress_bonus": progress_bonus,
        "terminal_bonus": terminal_bonus,
        "unclipped_total_reward": unclipped_total,
        "total_reward": clip_reward(unclipped_total),
    }


def apply_move_and_get_reward(game: GameEngine, response: dict) -> float:
    return calculate_reward_components(game, response)["total_reward"]
