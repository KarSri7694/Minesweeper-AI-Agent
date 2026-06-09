from __future__ import annotations

import argparse
import os

import uvicorn

from .api import create_app
from .cli import run_terminal_game
from .engine import GameManager
from .gui import run_gui_game


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Minesweeper app entrypoint.")
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--mine-density", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--spectate-game-id", default=None)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)

    mode = os.getenv("APP_MODE", "terminal").lower()
    manager = GameManager()

    if mode == "terminal":
        game = manager.create_game(
            width=args.width,
            height=args.height,
            mine_density=args.mine_density,
            seed=args.seed,
        )
        run_terminal_game(game)
        return

    if mode == "api":
        uvicorn.run(
            create_app(manager),
            host=args.host,
            port=args.port,
        )
        return

    if mode == "gui":
        run_gui_game(
            manager=manager,
            width=args.width,
            height=args.height,
            mine_density=args.mine_density,
            seed=args.seed,
            spectate_game_id=args.spectate_game_id or os.getenv("SPECTATE_GAME_ID"),
            api_base_url=args.api_base_url,
        )
        return

    raise ValueError("APP_MODE must be one of: terminal, api, gui.")
