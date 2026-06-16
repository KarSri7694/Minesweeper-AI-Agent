import requests
import json
from openai import AsyncOpenAI
from pathlib import Path
import asyncio
import time
import ast
base_url = "http://localhost:8080/"
api_url_v1 = base_url + "v1/"

game_url = "http://localhost:8000/"

MODEL_NAME = "Qwen-3-4B-Instruct-2507-Minesweeper-Agent"
def get_system_prompt():
    parent_dir = Path(__file__).parent.parent
    system_prompt = parent_dir / "system_prompt.md"
    with open(system_prompt, "r", encoding='utf-8') as f:
        return f.read()
    
def create_game():
    body= '''
    {
        "width": 12,
        "height": 12,
        "mine_density": 0.15,
        "seed": 7,
        "output_format": "compact"
    }
    '''
    json_data = json.loads(body)
    response = requests.post(game_url + "games/", json=json_data)
    if response.status_code == 200:
        print("Game created successfully!")
        game_id = response.json().get("game_id")
        print(f"Game ID: {game_id}, waiting 5 seconds")
        time.sleep(5)
        return game_id

def get_game_state(game_id):
    json_data = {
        "output_format": "compact"
    }
    response = requests.post(game_url + f"games/{game_id}/state", json=json_data)
    # print(response.json().get("board"))
    return response.json()
    
def make_move(game_id, action, x, y):
    body = {
        "action": action,
        "x": x,
        "y": y,
        "output_format": "compact"
    }
    response = requests.post(game_url + f"games/{game_id}/moves/", json=body)
    if response.status_code == 200:
        print("Move made successfully!")
        return response.json()
    else:
        print("Failed to make move.")
        print(response.text)
        return None

def connect_to_llm():
    llm = AsyncOpenAI(
        api_key="your-api-key",
        base_url=api_url_v1
    )
    return llm

async def get_llm_response(llm, system_prompt, user_prompt):
    
    response = await llm.chat.completions.create(
        model = MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True
    )
    # response = response.choices[0].message.content
    # print(f"LLM Response: {response}")
    return response

async def _consume_stream(response):
    assistant_text = ""
    async for chunk in response:
        delta = chunk.choices[0].delta

        # Standard content (the final answer text)
        if delta.content:
            print(delta.content, end="", flush=True)
            assistant_text += delta.content

        # Reasoning content (thinking / chain-of-thought)
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            print(f"\033[93m{reasoning}\033[0m", end="", flush=True)
    print()  # Newline after the stream is done
    return assistant_text

def parse_llm_response(response):
    response_json = ast.literal_eval(response)
    action = response_json.get("action")
    x = response_json.get("x")
    y = response_json.get("y")
    return action, x, y
    
async def main():
    game_id = create_game()
    game_state = get_game_state(game_id)
    llm = connect_to_llm()
    while True:
        response = await get_llm_response(llm, get_system_prompt(), f"Current game state: {game_state.get('board')}, Score: {game_state.get('score')}.")
        assistant_text = await _consume_stream(response)
        # print(f"Full LLM Response: {assistant_text}")
        
        try:
            # next_action = assistant_text.split("```json")[1].split("```")[0]
            # {'action': 'reveal', 'x': 8, 'y': 0}
            action, x, y = parse_llm_response(assistant_text)
        except json.decoder.JSONDecodeError as e:
            print(f"Failed to parse LLM response. Skipping turn. Error: {e}")
            continue
        except IndexError:
            next_action = assistant_text.strip()
            action, x, y = parse_llm_response(next_action)
            
        move_response = make_move(game_id, action, x, y)
        if move_response is None:
            break
        game_state = get_game_state(game_id)
        if game_state.get("status") in ["won", "lost"]:
            print(f"Game {game_state.get('status')}!, Final Score: {game_state.get('score')}")
            break
        

if __name__ == "__main__":
    asyncio.run(main())