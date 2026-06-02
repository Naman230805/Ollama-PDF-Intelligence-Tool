"""core/ollama_client.py — original version with temperature=0.05 hardcoded in ask_json"""

import json
import re
import requests
from typing import Optional


OLLAMA_BASE_URL = "http://localhost:11434"


def is_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def ask(
    prompt: str,
    model: str = "llama3.2",
    system: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    if not is_ollama_running():
        raise ConnectionError(
            "Ollama is not running. Start it with: ollama serve"
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"].strip()


def ask_json(
    prompt: str,
    model: str = "llama3.2",
    system: Optional[str] = None,
    retries: int = 2,
) -> dict:
    system_json = (system or "") + (
        "\n\nIMPORTANT: Respond ONLY with valid JSON. "
        "No markdown, no explanation, no code fences. Just raw JSON."
    )

    for attempt in range(retries + 1):
        raw = ask(prompt, model=model, system=system_json, temperature=0.05)
        try:
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            if attempt == retries:
                return {"error": "JSON parse failed", "raw_response": raw}

    return {}
