from __future__ import annotations

from .engine import GameEngine

NUMBER_REVEAL_REWARD = 0.35
ZERO_REVEAL_REWARD = 0.25
CORRECT_FLAG_REWARD = 0.45
WRONG_FLAG_PENALTY = -0.55
MINE_REVEAL_PENALTY = -3.0
WIN_REWARD = 1.5
CERTAINTY_BONUS = 0.25
FRONTIER_PROGRESS_BONUS = 0.10
INVALID_REVEAL_PENALTY = -0.5 #the model tries to reveal a tile that is already revealed
INVALID_FLAG_PENALTY = -0.5 #the model tries to flag a tile that is already revealed
INVALID_REPEAT_FLAG_PENALTY = -0.5 #the model tries to flag a tile that is already flagged
MIN_FINAL_REWARD = -2.0
MAX_FINAL_REWARD = 2.0


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


def _invalid_move_reward(penalty: float) -> dict[str, float]:
    clipped = clip_reward(penalty)
    return {
        "base_reward": penalty,
        "logic_bonus": 0.0,
        "progress_bonus": 0.0,
        "terminal_bonus": 0.0,
        "unclipped_total_reward": penalty,
        "total_reward": clipped,
    }


def calculate_reward_components(game: GameEngine, response: dict) -> dict[str, float]:
    """Return GRPO reward shaping without changing gameplay scoring rules."""
    action = response["action"]
    x = response["x"]
    y = response["y"]
    target_tile = game.get_tile(x, y)

    if action == "reveal" and target_tile.is_revealed:
        return _invalid_move_reward(INVALID_REVEAL_PENALTY)
    if action == "flag" and target_tile.is_revealed:
        return _invalid_move_reward(INVALID_FLAG_PENALTY)
    if action == "flag" and target_tile.is_flagged:
        return _invalid_move_reward(INVALID_REPEAT_FLAG_PENALTY)

    forced_safe, forced_mines = _find_forced_moves(game)
    target = (x, y)

    logic_bonus = 0.0
    if action == "reveal":
        if target in forced_safe:
            logic_bonus += CERTAINTY_BONUS
    else:
        if target in forced_mines:
            logic_bonus += CERTAINTY_BONUS

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
