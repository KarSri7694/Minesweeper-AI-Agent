from __future__ import annotations

import json
import os
import sys

import requests


def get_base_url() -> str:
    return (
        os.getenv("OLLAMA_HOST")
        or os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_URL")
        or "http://127.0.0.1:11434"
    ).rstrip("/")


def get_model() -> str:
    return os.getenv("OLLAMA_MODEL", "minesweeper-grpo")


def get_timeout() -> int:
    return int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))


def main() -> int:
    base_url = get_base_url()
    model = get_model()
    timeout = get_timeout()

    print(f"Testing Ollama at: {base_url}")
    print(f"Expected model: {model}")

    tags_response = requests.get(
        f"{base_url}/api/tags",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=timeout,
    )
    tags_response.raise_for_status()
    tags_payload = tags_response.json()
    models = [item.get("name") for item in tags_payload.get("models", [])]
    print(f"Models found: {models}")

    if model not in models:
        print(f"Configured model '{model}' was not present in /api/tags.", file=sys.stderr)
        return 1

    generate_response = requests.post(
        f"{base_url}/api/generate",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
        json={
            "model": model,
            "system": "Return exactly one Minesweeper move as JSON only.",
            "prompt": "[['.', '.', '.'], ['.', '.', '.'], ['.', '.', '.']]",
            "stream": True,
            "think": False,
            "options": {"num_predict": 64},
        },
        timeout=timeout,
        stream=True,
    )
    generate_response.raise_for_status()

    chunks = []
    for line in generate_response.iter_lines(decode_unicode=True):
        if not line:
            continue
        payload = json.loads(line)
        if "response" in payload:
            chunks.append(payload["response"])
        if payload.get("done"):
            break

    text = "".join(chunks).strip()
    print(f"Generated text: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
