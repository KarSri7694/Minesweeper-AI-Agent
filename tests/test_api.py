from __future__ import annotations

import sys
from pathlib import Path
import unittest

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minesweeper.api import CreateGameRequest, MoveRequest, create_app


def get_endpoint(app, path: str, method: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"Endpoint {method} {path} not found.")


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.create_game_endpoint = get_endpoint(self.app, "/games", "POST")
        self.get_game_endpoint = get_endpoint(self.app, "/games/{game_id}", "GET")
        self.move_endpoint = get_endpoint(self.app, "/games/{game_id}/moves", "POST")

    def test_create_get_and_move_game(self) -> None:
        created = self.create_game_endpoint(
            CreateGameRequest(width=5, height=5, mine_density=0.15, seed=7)
        )
        self.assertEqual(created["status"], "in_progress")

        game_id = created["game_id"]
        fetched = self.get_game_endpoint(game_id)
        self.assertEqual(fetched["game_id"], game_id)

        moved = self.move_endpoint(
            game_id,
            MoveRequest(action="reveal", x=0, y=0),
        )
        self.assertEqual(moved["last_move"]["action"], "reveal")
        self.assertGreaterEqual(moved["score"], 1)

    def test_invalid_action_returns_400(self) -> None:
        created = self.create_game_endpoint(CreateGameRequest(width=5, height=5))
        game_id = created["game_id"]

        with self.assertRaises(HTTPException) as context:
            self.move_endpoint(
                game_id,
                MoveRequest(action="jump", x=0, y=0),
            )
        self.assertEqual(context.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
