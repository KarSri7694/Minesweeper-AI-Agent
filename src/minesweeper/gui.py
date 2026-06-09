from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pygame

from .engine import GameEngine, GameManager, GameStatus


BACKGROUND = (245, 240, 228)
GRID_BG = (219, 209, 192)
HIDDEN_TILE = (126, 154, 123)
REVEALED_TILE = (235, 231, 220)
FLAGGED_TILE = (196, 107, 78)
MINE_TILE = (53, 62, 74)
TEXT = (34, 40, 49)
SUBTEXT = (92, 99, 112)
GRID_LINE = (162, 149, 129)
NUMBER_COLORS = {
    1: (52, 99, 181),
    2: (44, 122, 72),
    3: (176, 72, 58),
    4: (98, 74, 147),
    5: (154, 93, 40),
    6: (52, 126, 136),
    7: (65, 65, 65),
    8: (112, 112, 112),
}

PADDING = 18
HUD_HEIGHT = 90
FOOTER_HEIGHT = 42
MIN_TILE_SIZE = 28
MAX_TILE_SIZE = 56


def fetch_remote_state(api_base_url: str, game_id: str) -> dict:
    normalized_base_url = api_base_url.rstrip("/")
    url = f"{normalized_base_url}/games/{game_id}?output_format=detailed"
    with urlopen(url, timeout=2.0) as response:
        return json.loads(response.read().decode("utf-8"))


def run_gui_game(
    manager: GameManager,
    width: int,
    height: int,
    mine_density: float,
    seed: int | None = None,
    spectate_game_id: str | None = None,
    api_base_url: str = "http://127.0.0.1:8000",
) -> None:
    pygame.init()
    pygame.display.set_caption("Minesweeper LLM")

    game = None
    current_state = None
    spectator_mode = spectate_game_id is not None
    last_fetch_at = 0
    fetch_interval_ms = 250

    if spectator_mode:
        try:
            current_state = fetch_remote_state(api_base_url, spectate_game_id)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            pygame.quit()
            raise RuntimeError(
                f"Unable to fetch game '{spectate_game_id}' from {api_base_url}: {exc}"
            ) from exc
        width = current_state["width"]
        height = current_state["height"]
    else:
        game = manager.create_game(width=width, height=height, mine_density=mine_density, seed=seed)
        current_state = game.visible_state()

    screen = _create_screen(width, height)
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("consolas", 28, bold=True)
    body_font = pygame.font.SysFont("consolas", 20)
    tile_size = _choose_tile_size(width, height)
    tile_font = pygame.font.SysFont("consolas", max(18, tile_size // 2), bold=True)

    board_origin_x = PADDING
    board_origin_y = HUD_HEIGHT
    if spectator_mode:
        message = f"Watching API game {spectate_game_id}. Refreshes automatically."
    else:
        message = "Left click to reveal. Right click to flag. Press R to restart."

    running = True
    while running:
        if spectator_mode:
            now = pygame.time.get_ticks()
            if now - last_fetch_at >= fetch_interval_ms:
                try:
                    current_state = fetch_remote_state(api_base_url, spectate_game_id)
                    last_fetch_at = now
                except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                    message = f"API sync failed: {exc}"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    continue
                if event.key == pygame.K_r:
                    if spectator_mode:
                        message = "Restart is disabled while spectating an API game."
                        continue
                    game = manager.create_game(
                        width=width,
                        height=height,
                        mine_density=mine_density,
                        seed=seed,
                    )
                    current_state = game.visible_state()
                    message = "New game started."
                    continue

            if spectator_mode:
                continue

            if event.type == pygame.MOUSEBUTTONDOWN:
                tile_pos = _mouse_to_tile(
                    event.pos,
                    board_origin_x,
                    board_origin_y,
                    tile_size,
                    width,
                    height,
                )
                if tile_pos is None:
                    continue
                x, y = tile_pos
                try:
                    if event.button == 1:
                        result = game.reveal(x, y)
                        message = f"{result.message} score {result.score_delta:+d}"
                    elif event.button == 3:
                        result = game.flag(x, y)
                        message = f"{result.message} score {result.score_delta:+d}"
                    current_state = game.visible_state()
                except ValueError as exc:
                    message = str(exc)

        _draw(
            screen=screen,
            state=current_state,
            tile_size=tile_size,
            board_origin_x=board_origin_x,
            board_origin_y=board_origin_y,
            title_font=title_font,
            body_font=body_font,
            tile_font=tile_font,
            message=message,
            spectator_mode=spectator_mode,
        )
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def _create_screen(width: int, height: int) -> pygame.Surface:
    tile_size = _choose_tile_size(width, height)
    board_width = width * tile_size
    board_height = height * tile_size
    screen_width = board_width + (PADDING * 2)
    screen_height = HUD_HEIGHT + board_height + FOOTER_HEIGHT + (PADDING * 2)
    return pygame.display.set_mode((screen_width, screen_height))


def _choose_tile_size(width: int, height: int) -> int:
    largest_dimension = max(width, height)
    tile_size = 640 // max(1, largest_dimension)
    return max(MIN_TILE_SIZE, min(MAX_TILE_SIZE, tile_size))


def _mouse_to_tile(
    mouse_pos: tuple[int, int],
    board_origin_x: int,
    board_origin_y: int,
    tile_size: int,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    mouse_x, mouse_y = mouse_pos
    local_x = mouse_x - board_origin_x
    local_y = mouse_y - board_origin_y
    if local_x < 0 or local_y < 0:
        return None
    x = local_x // tile_size
    y = local_y // tile_size
    if 0 <= x < width and 0 <= y < height:
        return int(x), int(y)
    return None


def _draw(
    screen: pygame.Surface,
    state: dict,
    tile_size: int,
    board_origin_x: int,
    board_origin_y: int,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    tile_font: pygame.font.Font,
    message: str,
    spectator_mode: bool,
) -> None:
    screen.fill(BACKGROUND)
    board_width = state["width"] * tile_size
    board_height = state["height"] * tile_size
    status_value = state["status"]

    title = title_font.render("Minesweeper LLM", True, TEXT)
    screen.blit(title, (PADDING, 18))

    status_text = (
        f"Status: {status_value}   Score: {state['score']}   Flags: {state['flagged_count']}/{state['mine_count']}"
    )
    screen.blit(body_font.render(status_text, True, TEXT), (PADDING, 52))

    if status_value == GameStatus.WON.value:
        message_color = NUMBER_COLORS[2]
    elif status_value == GameStatus.LOST.value:
        message_color = NUMBER_COLORS[3]
    else:
        message_color = SUBTEXT
    screen.blit(body_font.render(message, True, message_color), (PADDING, 74))

    pygame.draw.rect(
        screen,
        GRID_BG,
        pygame.Rect(board_origin_x, board_origin_y, board_width, board_height),
        border_radius=8,
    )

    for row in state["board"]:
        for tile in row:
            _draw_tile(
                screen=screen,
                tile=tile,
                tile_size=tile_size,
                board_origin_x=board_origin_x,
                board_origin_y=board_origin_y,
                tile_font=tile_font,
            )

    if spectator_mode:
        controls = "Watching API session | Auto refresh | Esc quit"
    else:
        controls = "Left click reveal | Right click flag | R restart | Esc quit"
    footer_y = board_origin_y + board_height + 14
    screen.blit(body_font.render(controls, True, SUBTEXT), (PADDING, footer_y))


def _draw_tile(
    screen: pygame.Surface,
    tile: dict,
    tile_size: int,
    board_origin_x: int,
    board_origin_y: int,
    tile_font: pygame.font.Font,
) -> None:
    x = board_origin_x + (tile["x"] * tile_size)
    y = board_origin_y + (tile["y"] * tile_size)
    rect = pygame.Rect(x, y, tile_size - 1, tile_size - 1)

    state = tile["state"]
    if state == "hidden":
        fill = HIDDEN_TILE
        label = ""
        label_color = TEXT
    elif state == "flagged":
        fill = FLAGGED_TILE
        label = "F"
        label_color = (255, 247, 233)
    elif state == "mine":
        fill = MINE_TILE
        label = "*"
        label_color = (255, 239, 214)
    else:
        fill = REVEALED_TILE
        adjacent = tile["adjacent_mines"]
        label = "" if not adjacent else str(adjacent)
        label_color = NUMBER_COLORS.get(adjacent, TEXT)

    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, GRID_LINE, rect, width=1, border_radius=6)

    if label:
        label_surface = tile_font.render(label, True, label_color)
        label_rect = label_surface.get_rect(center=rect.center)
        screen.blit(label_surface, label_rect)
