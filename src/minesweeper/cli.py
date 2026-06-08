from __future__ import annotations

from .engine import GameEngine


def render_board(game: GameEngine) -> str:
    lines = []
    header = "   " + " ".join(f"{x:2d}" for x in range(game.config.width))
    lines.append(header)
    for y, row in enumerate(game.visible_state()["board"]):
        cells = []
        for tile in row:
            state = tile["state"]
            if state == "hidden":
                symbol = "."
            elif state == "flagged":
                symbol = "F"
            elif state == "mine":
                symbol = "*"
            else:
                adjacent = tile["adjacent_mines"]
                symbol = str(adjacent) if adjacent else " "
            cells.append(f"{symbol:>2}")
        lines.append(f"{y:2d} " + " ".join(cells))
    lines.append(
        f"score={game.score} status={game.status.value} flags={game.flagged_count}/{game.mine_count}"
    )
    return "\n".join(lines)


def run_terminal_game(game: GameEngine) -> None:
    print("Commands: reveal x y | flag x y | quit")
    while True:
        print(render_board(game))
        if game.status.value != "in_progress":
            print(f"Game finished: {game.status.value} ({game.end_reason})")
            break

        raw = input("> ").strip()
        if not raw:
            continue
        if raw.lower() == "quit":
            break

        parts = raw.split()
        if len(parts) != 3:
            print("Expected: reveal x y OR flag x y")
            continue

        action, x_str, y_str = parts
        try:
            x = int(x_str)
            y = int(y_str)
        except ValueError:
            print("Coordinates must be integers.")
            continue

        try:
            if action == "reveal":
                result = game.reveal(x, y)
            elif action == "flag":
                result = game.flag(x, y)
            else:
                print("Action must be 'reveal' or 'flag'.")
                continue
            print(f"{result.message} score_delta={result.score_delta}")
        except ValueError as exc:
            print(str(exc))
