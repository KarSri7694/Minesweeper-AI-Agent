
import numpy as np
import random
import json
import os
import sys
import argparse
from typing import List, Tuple, Dict, Set, Optional, FrozenSet
from dataclasses import dataclass

# Constant scalar definitions for state mappings
HIDDEN = 0
REVEALED = 1
FLAGGED = 2
MINE = -1


def to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


class Board:
    """
    Encapsulates the two-dimensional Minesweeper environment state, managing lattice
    topologies, discrete mine distributions, and localized cellular state transitions.
    """
    def __init__(self, rows: int, cols: int, num_mines: int):
        """
        Initializes an empty topological configuration.
        Mines are not strictly populated until programmatic assignment, standard stochastic
        generation, or upon the execution of the first safe-guaranteed reveal action.

        Args:
            rows (int): The vertical dimensional bound of the lattice.
            cols (int): The horizontal dimensional bound of the lattice.
            num_mines (int): The absolute quantity of latent mines to be distributed.
        """
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines

        # The latent grid encapsulates absolute truth: 0-8 for adjacent mine summation, or MINE (-1).
        self.grid = np.zeros((rows, cols), dtype=int)

        # The observable state encapsulates agent interaction constraints.
        self.state = np.full((rows, cols), HIDDEN, dtype=int)

        self.game_over = False
        self.is_won = False
        self.first_click = True
        self.score = 0
        self.end_reason: Optional[str] = None
        self.last_score_delta = 0

    @classmethod
    def from_config(cls, grid: np.ndarray, mines_positions: Set[Tuple[int, int]]) -> 'Board':
        """
        Factory instantiation method designed to construct a predetermined, explicit board state.
        This mechanism is critical for configuring exact environmental matrices during dataset
        generation, reproducibility testing, and algorithmic bottleneck evaluation.

        Args:
            grid (np.ndarray): A fully populated 2D integer matrix denoting the latent values.
            mines_positions (Set[Tuple[int, int]]): The explicit coordinate mapping of all mines.

        Returns:
            Board: A fully initialized environment matching the input parameters.
        """
        rows, cols = grid.shape
        num_mines = len(mines_positions)
        board = cls(rows, cols, num_mines)
        board.grid = np.copy(grid)
        board.first_click = False
        return board

    def generate_mines(self, safe_r: int, safe_c: int) -> None:
        """
        Stochastically populates latent mines across the lattice structure.
        Ensures the initial interaction coordinate itself remains devoid of mines,
        matching the current game rules.

        Args:
            safe_r (int): The row coordinate of the initial safe constraint.
            safe_c (int): The column coordinate of the initial safe constraint.
        """
        all_positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        valid_positions = [p for p in all_positions if p != (safe_r, safe_c)]

        # Standard stochastic sampling for mine distribution
        mine_positions = random.sample(valid_positions, self.num_mines)
        for r, c in mine_positions:
            self.grid[r, c] = MINE

        self._calculate_numbers()

    def _calculate_numbers(self) -> None:
        """
        Iterates over the entire spatial topology to calculate the precise summation
        of adjacent mines for all non-mine vertices based on the Moore neighborhood.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == MINE:
                    continue
                mines_count = sum(1 for nr, nc in self.get_neighbors(r, c)
                                  if self.grid[nr, nc] == MINE)
                self.grid[r, c] = mines_count

    def get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """
        Yields all valid, in-bound coordinates within the Moore neighborhood of a target cell.

        Args:
            r (int): Target row.
            c (int): Target column.

        Returns:
            List[Tuple[int, int]]: A collection of adjacent coordinate tuples.
        """
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors

    def reveal(self, r: int, c: int) -> bool:
        """
        Executes a deterministic state transition, morphing a cell from HIDDEN to REVEALED.
        If the revealed coordinate constitutes a zero-value state, recursive cascading
        expansion triggers to maximize observable territory.

        Args:
            r (int): Action row coordinate.
            c (int): Action column coordinate.

        Returns:
            bool: False if the action precipitates a catastrophic mine detonation; True otherwise.
        """
        self.last_score_delta = 0

        if self.state[r, c] == FLAGGED:
            return True

        if self.state[r, c] != HIDDEN or self.game_over:
            return True

        if self.first_click:
            self.generate_mines(r, c)
            self.first_click = False

        revealed_before = np.count_nonzero(self.state == REVEALED)
        self.state[r, c] = REVEALED

        if self.grid[r, c] == MINE:
            self.game_over = True
            self.end_reason = "revealed_mine"
            return False

        if self.grid[r, c] == 0:
            for nr, nc in self.get_neighbors(r, c):
                if self.state[nr, nc] == HIDDEN:
                    self.reveal(nr, nc)

        revealed_after = np.count_nonzero(self.state == REVEALED)
        self.last_score_delta = revealed_after - revealed_before
        self.score += self.last_score_delta
        self._check_win_condition()
        return True

    def flag(self, r: int, c: int) -> None:
        """
        Toggles an informational flag constraint on the specified coordinate.
        Flags are utilized by the solver logic to track known mine distributions.
        """
        self.last_score_delta = 0

        if self.game_over:
            return

        if self.state[r, c] == HIDDEN:
            self.state[r, c] = FLAGGED
            if self.grid[r, c] == MINE:
                self.last_score_delta = 2
            else:
                self.last_score_delta = -2
            self.score += self.last_score_delta
        elif self.state[r, c] == FLAGGED:
            self.state[r, c] = HIDDEN
        self._check_win_condition()

    def _check_win_condition(self) -> None:
        """
        Evaluates the global state tensor to determine if the terminal victory
        condition has been satisfied (all non-mine cells are formally revealed).
        """
        all_safe_revealed = True
        all_mines_flagged = True
        no_false_flags = True

        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == MINE and self.state[r, c] != FLAGGED:
                    all_mines_flagged = False
                if self.grid[r, c] != MINE and self.state[r, c] != REVEALED:
                    all_safe_revealed = False
                if self.grid[r, c] != MINE and self.state[r, c] == FLAGGED:
                    no_false_flags = False

        if all_safe_revealed and all_mines_flagged and no_false_flags:
            self.is_won = True
            self.game_over = True
            self.end_reason = "all_mines_flagged"
            self.last_score_delta += 50
            self.score += 50

    def is_solved(self) -> bool:
        """Returns the terminal Boolean victory status of the cellular environment."""
        return self.is_won

    def clone(self) -> 'Board':
        """
        Constructs a mathematically identical, unlinked memory duplicate of the environment.
        This enables external algorithms to execute state-mutating forward simulations
        without irreversibly corrupting the ground-truth environment.
        """
        new_board = Board(self.rows, self.cols, self.num_mines)
        new_board.grid = np.copy(self.grid)
        new_board.state = np.copy(self.state)
        new_board.game_over = self.game_over
        new_board.is_won = self.is_won
        new_board.first_click = self.first_click
        new_board.score = self.score
        new_board.end_reason = self.end_reason
        new_board.last_score_delta = self.last_score_delta
        return new_board

    def visible_state(self) -> Dict:
        board = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                row.append(self._serialize_tile(r, c))
            board.append(row)
        return {
            "status": "won" if self.is_won else "lost" if self.game_over else "in_progress",
            "width": self.cols,
            "height": self.rows,
            "mine_count": self.num_mines,
            "flagged_count": int(np.count_nonzero(self.state == FLAGGED)),
            "score": self.score,
            "end_reason": self.end_reason,
            "board": board,
        }

    def compact_state(self) -> Dict:
        board = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                row.append(self._compact_tile_value(r, c))
            board.append(row)
        return {
            "status": "won" if self.is_won else "lost" if self.game_over else "in_progress",
            "width": self.cols,
            "height": self.rows,
            "mine_count": self.num_mines,
            "flagged_count": int(np.count_nonzero(self.state == FLAGGED)),
            "score": self.score,
            "end_reason": self.end_reason,
            "board": board,
        }

    def _serialize_tile(self, r: int, c: int) -> Dict:
        if not self.game_over:
            if self.state[r, c] == FLAGGED:
                state = "flagged"
            elif self.state[r, c] == REVEALED:
                state = "revealed"
            else:
                state = "hidden"
            return {
                "x": c,
                "y": r,
                "state": state,
                "adjacent_mines": int(self.grid[r, c]) if self.state[r, c] == REVEALED else None,
            }

        terminal_state = "revealed" if self.state[r, c] == REVEALED else "hidden"
        if self.state[r, c] == FLAGGED:
            terminal_state = "flagged"
        if self.grid[r, c] == MINE and not self.is_won:
            terminal_state = "mine"
        return {
            "x": c,
            "y": r,
            "state": terminal_state,
            "adjacent_mines": None if self.grid[r, c] == MINE and self.state[r, c] != REVEALED else int(self.grid[r, c]),
            "is_mine": bool(self.grid[r, c] == MINE),
        }

    def _compact_tile_value(self, r: int, c: int) -> str:
        if self.state[r, c] == FLAGGED:
            return "F"
        if self.game_over and not self.is_won and self.grid[r, c] == MINE:
            return "B"
        if self.state[r, c] != REVEALED:
            return "."
        if self.grid[r, c] == 0:
            return "_"
        return str(int(self.grid[r, c]))


# ---------------------------------------------------------------------------
# Part 2: Deterministic Constraint Satisfaction Optimization
# ---------------------------------------------------------------------------

@dataclass
class SolverMove:
    """
    Standardized payload structure encapsulating actionable heuristic derivations.
    Transfers the resolution intent to the underlying environment execution logic.
    """
    type: str  # Strictly enumerated as 'REVEAL' or 'FLAG'
    cells: List[Tuple[int, int]]


@dataclass
class Constraint:
    """
    Mathematical abstraction of a localized linear constraint equation.
    Maps a specific set of unrevealed cellular coordinates to a targeted mine integer.
    """
    cells: FrozenSet[Tuple[int, int]]
    count: int

    def __hash__(self):
        """Hash generation to facilitate high-speed Set intersections."""
        return hash((self.cells, self.count))


class Solver:
    """
    Deterministic inference engine built upon boolean constraint satisfaction
    methodologies. Implements recursive single-cell heuristics combined with
    advanced subset-reduction linear algebra.
    """
    def __init__(self, board: Board):
        self.board = board
        self.stuck = False

    def solve_step(self) -> Optional[SolverMove]:
        """
        Executes a single deterministically guaranteed mathematical resolution.
        Iterates exhaustively through base single-cell logic, followed by
        intersecting subset derivations, and finally global limits.

        Returns:
            Optional[SolverMove]: The defined categorical action, or None if the
            board requires probabilistic guessing.
        """
        # Step 1: Base linear single-cell propositional logic
        for r in range(self.board.rows):
            for c in range(self.board.cols):
                if self.board.state[r, c] == REVEALED and self.board.grid[r, c] > 0:
                    neighbors = self.board.get_neighbors(r, c)
                    hidden = [n for n in neighbors if self.board.state[n[0], n[1]] == HIDDEN]
                    flagged = [n for n in neighbors if self.board.state[n[0], n[1]] == FLAGGED]

                    value = self.board.grid[r, c]
                    remaining_mines = value - len(flagged)

                    if len(hidden) > 0:
                        # Boundary resolution: All variables are definitively safe
                        if remaining_mines == 0:
                            return SolverMove(type='REVEAL', cells=hidden)
                        # Boundary resolution: All variables are definitively mines
                        elif len(hidden) == remaining_mines:
                            return SolverMove(type='FLAG', cells=hidden)

        # Step 2: Extraction of active frontiers for subset algebra
        constraints = self._extract_constraints()

        # Step 3: Global Topological Limit Integration
        total_flags = np.count_nonzero(self.board.state == FLAGGED)
        all_hidden = frozenset(
            (r, c) for r in range(self.board.rows) for c in range(self.board.cols)
            if self.board.state[r, c] == HIDDEN
        )
        if all_hidden:
            global_count = self.board.num_mines - total_flags
            constraints.add(Constraint(cells=all_hidden, count=global_count))

        # Iterative Sub-matrix Reduction Loop
        # Guards: MAX_CONSTRAINTS caps set growth (prevents O(N²) explosion as derived
        # constraints accumulate); MAX_REDUCTION_ITERS caps full passes (Minesweeper
        # puzzles resolve within 2–3 iterations in practice or not at all).
        MAX_CONSTRAINTS = 500
        MAX_REDUCTION_ITERS = 10
        new_constraints = set(constraints)
        progress = True
        iteration = 0
        while progress and iteration < MAX_REDUCTION_ITERS:
            progress = False
            iteration += 1
            current_constraints = list(new_constraints)
            for i in range(len(current_constraints)):
                for j in range(len(current_constraints)):
                    if i == j:
                        continue
                    A = current_constraints[i]
                    B = current_constraints[j]

                    # Identifies strict logical sub-groupings
                    if A.cells and A.cells.issubset(B.cells) and A.cells != B.cells:
                        derived_cells = B.cells - A.cells
                        derived_count = B.count - A.count
                        derived_constraint = Constraint(cells=derived_cells, count=derived_count)

                        if derived_constraint not in new_constraints:
                            new_constraints.add(derived_constraint)
                            progress = True

                            # Evaluates if the newly synthesized constraint achieves boundary logic
                            if derived_count == 0:
                                return SolverMove(type='REVEAL', cells=list(derived_cells))
                            elif len(derived_cells) == derived_count:
                                return SolverMove(type='FLAG', cells=list(derived_cells))

                        # Stop adding if constraint set has grown too large
                        if len(new_constraints) >= MAX_CONSTRAINTS:
                            progress = False
                            break
                if not progress:
                    break

        # Flag indicating the limits of deterministic linear reduction have been reached
        self.stuck = True
        return None

    def _extract_constraints(self) -> Set[Constraint]:
        """
        Scans the currently exposed topological frontier to map localized constraints.

        Returns:
            Set[Constraint]: A unique set of mathematical boundaries defined by the frontier.
        """
        constraints = set()
        for r in range(self.board.rows):
            for c in range(self.board.cols):
                if self.board.state[r, c] == REVEALED and self.board.grid[r, c] > 0:
                    neighbors = self.board.get_neighbors(r, c)
                    hidden = frozenset(n for n in neighbors if self.board.state[n[0], n[1]] == HIDDEN)
                    flagged_count = sum(1 for n in neighbors if self.board.state[n[0], n[1]] == FLAGGED)
                    if hidden:
                        count = self.board.grid[r, c] - flagged_count
                        constraints.add(Constraint(cells=hidden, count=count))
        return constraints

    def solve_all(self) -> List[SolverMove]:
        """
        Orchestrates an uninterrupted execution of continuous deterministic inference
        until absolute resolution is achieved or mathematical exhaustion occurs.

        Returns:
            List[SolverMove]: An ordered sequential log of all executed boolean deductions.
        """
        moves = []
        self.stuck = False
        while not self.board.game_over:
            move = self.solve_step()
            if move is None:
                break
            moves.append(move)
            if move.type == 'REVEAL':
                for r, c in move.cells:
                    self.board.reveal(r, c)
            elif move.type == 'FLAG':
                for r, c in move.cells:
                    self.board.flag(r, c)
        return moves

    def is_stuck(self) -> bool:
        """Returns the algorithmic exhaustion state of the subset deduction loop."""
        return self.stuck


# ---------------------------------------------------------------------------
# Part 3: Probabilistic Modeling and Combinatorial Enumeration
# ---------------------------------------------------------------------------

class ProbabilisticSolver:
    """
    Fallback inference architecture deploying heavily optimized combinatorial
    backtracking to ascertain exact fractional probabilities across interconnected
    and ambiguous topological sub-structures.
    """
    def __init__(self, board: Board):
        self.board = board
        # Implemented to prevent #P-Complete exponential explosion.
        self.MAX_CONFIGS = 10000

    def calculate_probabilities(self) -> Dict[Tuple[int, int], float]:
        """
        Analyzes the macroscopic grid to derive a normalized probability vector
        mapping for every unresolved coordinate. Separates the environment into
        a localized active frontier and a detached unconstrained void.

        Returns:
            Dict[Tuple[int, int], float]: A dictionary mapping spatial coordinates
            to floating-point probability representations bound within [0.0, 1.0].
        """
        solver = Solver(self.board)
        constraints = list(solver._extract_constraints())

        constrained_cells = set()
        for const in constraints:
            constrained_cells.update(const.cells)
        constrained_cells_list = list(constrained_cells)

        valid_configs: List[Dict[Tuple[int, int], int]] = []
        total_flags = np.count_nonzero(self.board.state == FLAGGED)
        remaining_mines_total = self.board.num_mines - total_flags

        # Hard cap on total nodes explored to guarantee termination on dense frontiers.
        # MAX_CONFIGS caps valid configs found; max_nodes caps tree traversal independent
        # of how many valid configs exist — critical for large, heavily-constrained frontiers.
        MAX_NODES = self.MAX_CONFIGS * 5
        nodes_visited = [0]

        def backtrack(index: int, current_assignment: Dict[Tuple[int, int], int]):
            """
            Recursive Depth-First traversing generator. Dynamically builds constraint
            satisfaction mappings and prunes computationally unviable branches.
            """
            if len(valid_configs) >= self.MAX_CONFIGS or nodes_visited[0] >= MAX_NODES:
                return

            nodes_visited[0] += 1

            if index == len(constrained_cells_list):
                mines_placed = sum(current_assignment.values())
                if mines_placed <= remaining_mines_total:
                    valid_configs.append(current_assignment.copy())
                return

            cell = constrained_cells_list[index]

            for assumption in [0, 1]:
                current_assignment[cell] = assumption

                is_valid = True
                for const in constraints:
                    if cell in const.cells:
                        assigned_mines = sum(current_assignment.get(c, 0) for c in const.cells if c in current_assignment)
                        unassigned_count = sum(1 for c in const.cells if c not in current_assignment)

                        # MRV Sub-branch Pruning Logic
                        if assigned_mines > const.count:
                            is_valid = False
                            break
                        if assigned_mines + unassigned_count < const.count:
                            is_valid = False
                            break

                if is_valid:
                    backtrack(index + 1, current_assignment)

            del current_assignment[cell]

        if constrained_cells_list:
            backtrack(0, {})

        probabilities: Dict[Tuple[int, int], float] = {}
        total_valid = len(valid_configs)

        # Empirical frontier probability generation
        if total_valid > 0:
            for cell in constrained_cells_list:
                mine_count = sum(config[cell] for config in valid_configs)
                probabilities[cell] = mine_count / total_valid
        else:
            total_valid = 1
            for cell in constrained_cells_list:
                probabilities[cell] = 0.5

        # Subtractive logic defining unconstrained binomial probabilities
        if total_valid > 0 and valid_configs:
            avg_mines_in_frontier = sum(sum(cfg.values()) for cfg in valid_configs) / total_valid
        else:
            avg_mines_in_frontier = 0

        unassigned_mines_global = remaining_mines_total - avg_mines_in_frontier

        all_hidden_cells = [(r, c) for r in range(self.board.rows) for c in range(self.board.cols)
                            if self.board.state[r, c] == HIDDEN]
        unconstrained_cells = [c for c in all_hidden_cells if c not in constrained_cells]

        global_prob = 0.0
        if len(unconstrained_cells) > 0:
            global_prob = unassigned_mines_global / len(unconstrained_cells)
            global_prob = max(0.0, min(1.0, global_prob))

        for cell in unconstrained_cells:
            probabilities[cell] = global_prob

        return probabilities

    def best_guess(self) -> Tuple[int, int]:
        """
        Parses the probability vector to extract the optimum action. Implements an
        advanced geometric tie-breaker prioritizing central matrix coordinates to
        catalyze multi-directional frontier expansion upon successful survival.

        Returns:
            Tuple[int, int]: The ideal geometric coordinate for a stochastic action.
        """
        probs = self.calculate_probabilities()
        if not probs:
            return (-1, -1)

        center_r, center_c = self.board.rows / 2.0, self.board.cols / 2.0

        best_cell = None
        min_prob = float('inf')
        min_dist = float('inf')

        for (r, c), p in probs.items():
            dist = (r - center_r)**2 + (c - center_c)**2
            if p < min_prob - 1e-6:
                min_prob = p
                min_dist = dist
                best_cell = (r, c)
            elif abs(p - min_prob) <= 1e-6:
                if dist < min_dist:
                    min_dist = dist
                    best_cell = (r, c)

        return best_cell


# ---------------------------------------------------------------------------
# Part 4: Markov Decision Process and RL Dataset Orchestration
# ---------------------------------------------------------------------------

class DatasetGenerator:
    """
    Automated environment orchestrator responsible for engineering dense RL training
    matrices. Formats deterministic constraint boundaries alongside stochastic probability
    curves into localized JSONL sequences.
    """
    def __init__(self, rows: int, cols: int, num_mines: int, output_dir: str):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.output_dir = output_dir

    def serialize_action(self, action: str, row: int, col: int) -> Dict:
        return {
            "action": action,
            "x": col,
            "y": row,
        }

    def generate_state_tensor(self, board: Board, probabilities: Dict[Tuple[int, int], float]) -> np.ndarray:
        """
        Projects the elementary state board arrays into computationally comprehensive,
        high-dimensional 5-channel matrices structured for multi-layer perception processing.

        Returns:
            np.ndarray: A scaled, normalized matrix of dimension (Rows, Cols, 5).
        """
        tensor = np.zeros((self.rows, self.cols, 5), dtype=np.float32)

        solver = Solver(board)
        constraints = solver._extract_constraints()
        constrained_set = set()
        for const in constraints:
            constrained_set.update(const.cells)

        for r in range(self.rows):
            for c in range(self.cols):
                # Channel 0: Revealed binary
                if board.state[r, c] == REVEALED:
                    tensor[r, c, 0] = 1.0

                # Channel 1: Flagged binary
                if board.state[r, c] == FLAGGED:
                    tensor[r, c, 1] = 1.0

                # Channel 2: Normalized observed value
                if board.state[r, c] == REVEALED and board.grid[r, c] > 0:
                    tensor[r, c, 2] = board.grid[r, c] / 8.0

                # Channel 3: Constrained frontier binary
                if (r, c) in constrained_set:
                    tensor[r, c, 3] = 1.0

                # Channel 4: Backtracking probability
                if board.state[r, c] == HIDDEN:
                    tensor[r, c, 4] = probabilities.get((r, c), 0.0)

        return tensor

    def play_episode(self, game_id: int) -> Tuple[List[Dict], Dict]:
        """
        Generates simulated episodes traversing from uninitialized zero states to
        definitive resolution or destruction. Accurately compiles comprehensive MDP
        transition dicts representing formal (State, Action, Reward, State_Prime).

        Performance optimization: probability computation is deferred and cached.
        For deterministic moves the expensive backtracking enumeration is skipped
        entirely — probabilities are only computed when the solver is stuck and a
        probabilistic guess is required.
        """
        board = Board(self.rows, self.cols, self.num_mines)
        transitions: List[Dict] = []

        start_r, start_c = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
        board.reveal(start_r, start_c)

        step_count = 0
        deterministic_moves = 0
        probabilistic_moves = 0

        # Cache: reuse next_state probs as current state probs on the following iteration
        cached_probs: Optional[Dict[Tuple[int, int], float]] = None

        while not board.game_over:
            state_before_compact = board.compact_state()
            solver = Solver(board)
            move = solver.solve_step()

            action_r, action_c = -1, -1
            move_type = "unknown"
            action_prob = 0.0
            api_action = "reveal"

            # Determine probabilities: only compute when needed
            if move is not None:
                # Deterministic move — use cached probs if available, else empty dict
                probs = cached_probs if cached_probs is not None else {}
                move_type = "deterministic"
                if move.type == 'REVEAL':
                    action_r, action_c = move.cells[0]
                    action_prob = 0.0
                    api_action = "reveal"
                    board.reveal(action_r, action_c)
                elif move.type == 'FLAG':
                    action_r, action_c = move.cells[0]
                    action_prob = 1.0
                    api_action = "flag"
                    board.flag(action_r, action_c)
                deterministic_moves += 1
            else:
                # Probabilistic move — must compute probabilities
                move_type = "probabilistic"
                prob_solver = ProbabilisticSolver(board)
                probs = prob_solver.calculate_probabilities()

                # Select best guess from the already-computed probabilities
                if not probs:
                    break
                center_r, center_c = board.rows / 2.0, board.cols / 2.0
                best_cell = None
                min_prob = float('inf')
                min_dist = float('inf')
                for (r, c), p in probs.items():
                    dist = (r - center_r)**2 + (c - center_c)**2
                    if p < min_prob - 1e-6:
                        min_prob = p
                        min_dist = dist
                        best_cell = (r, c)
                    elif abs(p - min_prob) <= 1e-6:
                        if dist < min_dist:
                            min_dist = dist
                            best_cell = (r, c)

                if best_cell is None:
                    break
                action_r, action_c = best_cell
                action_prob = probs.get((action_r, action_c), 0.5)
                api_action = "reveal"
                board.reveal(action_r, action_c)
                probabilistic_moves += 1

            step_count += 1

            # Compute next-state probabilities only on probabilistic transitions.
            # For deterministic steps, channel 4 stays 0 — semantically correct because
            # the solver has a guaranteed-safe action, so probability guidance is irrelevant.
            if not board.game_over and move_type == "probabilistic":
                next_probs = ProbabilisticSolver(board).calculate_probabilities()
                cached_probs = next_probs
            elif board.game_over:
                next_probs = {}
                cached_probs = None
            else:
                # Deterministic step: propagate cached probs, don't recompute
                next_probs = {}
                # cached_probs stays unchanged for next iteration
            state_after_compact = board.compact_state()

            transition = {
                "compact_board_before": state_before_compact["board"],
                "compact_board_after": state_after_compact["board"],
                "action": self.serialize_action(api_action, action_r, action_c),
                "score": state_after_compact["score"],
                "metadata": {
                    "game_id": game_id,
                    "step": step_count,
                    "current_state": state_after_compact["status"],
                    "move_type": move_type,
                    "mine_probability_at_action": action_prob,
                    "board_size": [self.rows, self.cols],
                    "mine_count": self.num_mines,
                    "output_format": "compact",
                }
            }
            transitions.append(transition)

        result = "win" if board.is_won else "loss"
        summary = {
            "game_id": game_id,
            "result": result,
            "steps": step_count,
            "deterministic_moves": deterministic_moves,
            "probabilistic_moves": probabilistic_moves,
            "board_size": [self.rows, self.cols],
            "mine_count": self.num_mines,
            "final_score": board.score,
        }

        return transitions, summary


# ---------------------------------------------------------------------------
# Part 5: Execution Topologies and System Orchestration
# ---------------------------------------------------------------------------

def main():
    """
    Extensible execution gateway formatting hyperparameter commands. Automates
    large-scale sequence runs mapping synthetic dataset outputs across standardized JSONL structures.
    """
    parser = argparse.ArgumentParser(description="Computational Minesweeper RL Matrix Generator")
    parser.add_argument("--rows", type=int, default=15, help="Vertical geometric constraint")
    parser.add_argument("--cols", type=int, default=15, help="Horizontal geometric constraint")
    parser.add_argument("--mines", type=int, default=35, help="Latent variable volume")
    parser.add_argument("--games", type=int, default=1000, help="Iteration limit for episode sequences")
    parser.add_argument("--output", type=str, default="./dataset", help="Target output structural directory")
    parser.add_argument("--filename", type=str, default="minesweeper_dataset.jsonl", help="Base filename for output sequences")
    parser.add_argument("--seed", type=int, default=None, help="Global stochastic entropy lock")

    # Internal bypass for notebook/module environments without stripping strict logic.
    if 'ipykernel' in sys.modules or len(sys.argv) == 1:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)
    transitions_path = os.path.join(args.output, args.filename or "minesweeper_dataset.jsonl")
    summaries_path = os.path.join(args.output, "games_summary.jsonl")

    generator = DatasetGenerator(args.rows, args.cols, args.mines, args.output)

    wins = 0
    total_steps = 0
    total_det = 0
    total_prob = 0

    print(f"Executing Dataset Synthesis -> Games: {args.games} | Geometry: {args.rows}x{args.cols} | Target Mines: {args.mines}")

    with open(transitions_path, 'w') as f_trans, open(summaries_path, 'w') as f_sum:
        for i in range(args.games):
            transitions, summary = generator.play_episode(i)

            for t in transitions:
                f_trans.write(json.dumps(to_jsonable(t)) + "\n")
            f_sum.write(json.dumps(to_jsonable(summary)) + "\n")

            # Flush after every game so progress is always visible on disk
            # and not lost if the process is interrupted mid-run.
            f_trans.flush()
            f_sum.flush()

            if summary["result"] == "win":
                wins += 1
            total_steps += summary["steps"]
            total_det += summary["deterministic_moves"]
            total_prob += summary["probabilistic_moves"]

            if (i + 1) % max(1, (args.games // 10)) == 0:
                print(f"Sequence Log: {i + 1} / {args.games} Epochs Processed.", flush=True)

    total_moves = total_det + total_prob
    if total_moves == 0:
        total_moves = 1

    print("\n--- Empirical Subsystem Generation Overview ---")
    print(f"Absolute Resolution Rate (Win): {(wins / args.games) * 100:.2f}%")
    print(f"Average Action Cycles/Episode:  {total_steps / args.games:.2f}")
    print(f"Algorithmic Deterministic Mass: {(total_det / total_moves) * 100:.2f}%")
    print(f"Stochastic Uncertainty Yield:   {(total_prob / total_moves) * 100:.2f}%")


def demonstrate_system():
    """
    Targeted procedural verification establishing an explicit, highly constrained
    topological setup to mathematically validate recursive sub-system transitions.
    """
    print("\n--- Targeted Diagnostic Validation Run ---")

    # 1. Structural instantiation of a mathematically precarious framework utilizing the config builder.
    synthetic_grid = np.zeros((9, 9), dtype=int)
    synthetic_mines = {(1, 1), (1, 3)}

    board = Board.from_config(synthetic_grid, synthetic_mines)
    board.grid[1, 1] = MINE
    board.grid[1, 3] = MINE
    board._calculate_numbers()

    # 2. Exposing strategic edge components to explicitly force deterministic solver interaction.
    board.state[0, 0] = REVEALED
    board.state[0, 1] = REVEALED
    board.state[0, 2] = REVEALED
    board.flag(1, 1)

    solver = Solver(board)

    # 3. Processing exhaustive deterministic cascading
    print("Initiating Constraint Reduction Operations...")
    while True:
        move = solver.solve_step()
        if not move:
            break
        print(f"Heuristic Applied -> Operation: {move.type} | Target Matrices: {move.cells}")
        if move.type == 'REVEAL':
            for r, c in move.cells:
                board.reveal(r, c)
        elif move.type == 'FLAG':
            for r, c in move.cells:
                board.flag(r, c)

    # 4. Probabilistic fallback processing upon mathematical exhaustion.
    if solver.is_stuck():
        print("\nDeterministic Limits Reached. Engaging Probabilistic Enumerator Matrix...")
        prob_solver = ProbabilisticSolver(board)
        probs = prob_solver.calculate_probabilities()

        print("Sub-Matrix Mathematical Bounds (Frontier Variables):")
        sample_count = 0
        for coord, probability in probs.items():
            if sample_count >= 5:
                break
            print(f"  Coordinate Scalar {coord} -> Exact Fallback Ratio P(X) = {probability:.4f}")
            sample_count += 1

        best_coord = prob_solver.best_guess()
        if best_coord != (-1, -1):
            print(f"\nMinimum Risk Actuation Index -> Vector {best_coord} (Calculated Deviation: {probs[best_coord]:.4f})")

    print("\nExecuting Subsystem Stress Test (100 Successive Validations)...")
    os.makedirs("./diagnostic_dataset", exist_ok=True)
    generator = DatasetGenerator(9, 9, 10, "./diagnostic_dataset")
    wins = 0
    total_steps, total_det, total_prob = 0, 0, 0
    for i in range(100):
        _, summary = generator.play_episode(i)
        if summary["result"] == "win":
            wins += 1
        total_steps += summary["steps"]
        total_det += summary["deterministic_moves"]
        total_prob += summary["probabilistic_moves"]

    total_moves = total_det + total_prob
    if total_moves == 0:
        total_moves = 1
    print(f"Stress Yield Statistics -> Win Rate: {wins}% | Mean Sequences: {total_steps/100:.2f} | Deterministic Density: {(total_det/total_moves)*100:.2f}%")


if __name__ == "__main__":
    main()
    demonstrate_system()
