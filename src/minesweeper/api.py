from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine import GameManager


class CreateGameRequest(BaseModel):
    width: int = Field(..., ge=2, le=50)
    height: int = Field(..., ge=2, le=50)
    mine_density: float = Field(0.15, gt=0, lt=1)
    seed: int | None = None


class MoveRequest(BaseModel):
    action: str
    x: int
    y: int


def create_app(manager: GameManager | None = None) -> FastAPI:
    app = FastAPI(title="Minesweeper AI API", version="1.0.0")
    game_manager = manager or GameManager()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/games")
    def create_game(payload: CreateGameRequest) -> dict:
        try:
            game = game_manager.create_game(
                width=payload.width,
                height=payload.height,
                mine_density=payload.mine_density,
                seed=payload.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return game.visible_state()

    @app.get("/games/{game_id}")
    def get_game(game_id: str) -> dict:
        try:
            game = game_manager.get_game(game_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return game.visible_state()

    @app.post("/games/{game_id}/moves")
    def play_move(game_id: str, payload: MoveRequest) -> dict:
        try:
            game = game_manager.get_game(game_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            if payload.action == "reveal":
                result = game.reveal(payload.x, payload.y)
            elif payload.action == "flag":
                result = game.flag(payload.x, payload.y)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Action must be either 'reveal' or 'flag'.",
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        response = game.visible_state()
        response["last_move"] = {
            "action": payload.action,
            "x": payload.x,
            "y": payload.y,
            "score_delta": result.score_delta,
            "message": result.message,
            "changed_tiles": result.changed_tiles,
        }
        return response

    return app
