from __future__ import annotations

from typing import Any

from .engine import GameEngine
from .solve_algo import Board, FLAGGED, HIDDEN, REVEALED, ProbabilisticSolver, ranked_solver_actions

NUMBER_REVEAL_REWARD = 0.35
ZERO_REVEAL_REWARD = 0.35
CORRECT_FLAG_REWARD = 0.8
WRONG_FLAG_PENALTY = -0.4
MINE_REVEAL_PENALTY = -3.0
WIN_REWARD = 1.5
DETERMINISTIC_MATCH_BONUS = 0.60
MISSED_DETERMINISTIC_PENALTY = -0.15
MAX_GUESS_QUALITY_BONUS = 0.20
MIN_GUESS_QUALITY_BONUS = -0.20
PROBABILISTIC_FLAG_PENALTY = -0.10
FRONTIER_PROGRESS_BONUS = 0.10
INVALID_REVEAL_PENALTY = -1.5 #the model tries to reveal a tile that is already revealed
INVALID_FLAG_PENALTY = -1.5 #the model tries to flag a tile that is already revealed
INVALID_REPEAT_FLAG_PENALTY = -1.5 #the model tries to flag a tile that is already flagged
MIN_FINAL_REWARD = -7.0
MAX_FINAL_REWARD = 7.0

RANK_BONUSES = {
    1: 0.10,
    2: 0.05,
    3: 0.02,
}


def _game_to_solver_board(game: GameEngine) -> Board:
    board = Board(game.config.height, game.config.width, game.mine_count)
    board.first_click = False
    board.game_over = game.status.value != "in_progress"
    board.is_won = game.status.value == "won"
    board.score = game.score
    board.end_reason = game.end_reason

    for tile in game.iter_tiles():
        board.grid[tile.y, tile.x] = -1 if tile.is_mine else tile.adjacent_mines
        if tile.is_revealed:
            board.state[tile.y, tile.x] = REVEALED
        elif tile.is_flagged:
            board.state[tile.y, tile.x] = FLAGGED
        else:
            board.state[tile.y, tile.x] = HIDDEN

    return board


def _guess_quality_bonus(probability: float) -> float:
    return max(
        MIN_GUESS_QUALITY_BONUS,
        min(MAX_GUESS_QUALITY_BONUS, MAX_GUESS_QUALITY_BONUS - (0.40 * probability)),
    )


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


def _evaluate_logic_signal(game: GameEngine, action: str, x: int, y: int) -> dict[str, Any]:
    solver_board = _game_to_solver_board(game)
    ranked_candidates = ranked_solver_actions(solver_board)
    deterministic_candidates = [candidate for candidate in ranked_candidates if candidate.move_type == "deterministic"]
    target = (action, y, x)

    if deterministic_candidates:
        deterministic_targets = {
            (candidate.action, candidate.row, candidate.col)
            for candidate in deterministic_candidates
        }
        matched = target in deterministic_targets
        return {
            "logic_bonus": DETERMINISTIC_MATCH_BONUS if matched else MISSED_DETERMINISTIC_PENALTY,
            "logic_state": "deterministic",
            "logic_match": "matched" if matched else "missed_deterministic",
            "mine_probability": 0.0 if matched and action == "reveal" else (1.0 if matched and action == "flag" else None),
            "candidate_rank": None,
            "deterministic_candidate_count": len(deterministic_candidates),
        }

    probabilities = ProbabilisticSolver(solver_board).calculate_probabilities()
    mine_probability = float(probabilities.get((y, x), 1.0))

    if action == "flag":
        return {
            "logic_bonus": PROBABILISTIC_FLAG_PENALTY,
            "logic_state": "probabilistic",
            "logic_match": "probabilistic_flag",
            "mine_probability": mine_probability,
            "candidate_rank": None,
            "deterministic_candidate_count": 0,
        }

    candidate_rank = None
    rank_bonus = 0.0
    for rank, candidate in enumerate(ranked_candidates, start=1):
        if candidate.action == action and candidate.row == y and candidate.col == x:
            candidate_rank = rank
            rank_bonus = RANK_BONUSES.get(rank, 0.0)
            break

    return {
        "logic_bonus": _guess_quality_bonus(mine_probability) + rank_bonus,
        "logic_state": "probabilistic",
        "logic_match": "ranked_guess" if candidate_rank is not None else "unranked_guess",
        "mine_probability": mine_probability,
        "candidate_rank": candidate_rank,
        "deterministic_candidate_count": 0,
    }


def calculate_reward_components(game: GameEngine, response: dict) -> dict[str, Any]:
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

    logic_evaluation = _evaluate_logic_signal(game, action, x, y)
    logic_bonus = float(logic_evaluation["logic_bonus"])

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
        "logic_state": logic_evaluation["logic_state"],
        "logic_match": logic_evaluation["logic_match"],
        "mine_probability": logic_evaluation["mine_probability"],
        "candidate_rank": logic_evaluation["candidate_rank"],
        "deterministic_candidate_count": logic_evaluation["deterministic_candidate_count"],
        "progress_bonus": progress_bonus,
        "terminal_bonus": terminal_bonus,
        "unclipped_total_reward": unclipped_total,
        "total_reward": clip_reward(unclipped_total),
    }


def apply_move_and_get_reward(game: GameEngine, response: dict) -> float:
    return calculate_reward_components(game, response)["total_reward"]
