"""Minimal client for TypeSafe AI's Jev (System One) model: POST /v1/systemone.

Jev doesn't generate text. It takes a state (text or JSON) plus named typed questions
(noul / choice / score) and returns typed answers with calibrated probabilities.
"""

import time

import httpx

from . import config


class JevError(RuntimeError):
    pass


def score(instructions: str, levels: list[str]) -> dict:
    """An ordered rubric, lowest level first (2-10 levels). The answer's `score` is the expected level."""
    return {"type": "score", "instructions": instructions, "criteria": levels}


def noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


def system_one(state, questions: dict, timeout: float | None = None, retries: int = 1) -> dict:
    """Returns the full response: {"model", "answers": {name: answer}, "usage"}."""
    if not config.TYPESAFE_API_KEY:
        raise JevError("TYPESAFE_API_KEY is not set. Add it to .env (see .env.example).")
    payload = {"model": config.JEV_MODEL, "state": state, "questions": questions}
    headers = {"Authorization": f"Bearer {config.TYPESAFE_API_KEY}"}
    url = f"{config.TYPESAFE_BASE_URL.rstrip('/')}/v1/systemone"
    for attempt in range(retries + 1):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=timeout or config.JEV_TIMEOUT)
        except httpx.TransportError as exc:  # connection problems and timeouts
            if attempt == retries:
                raise JevError(f"Jev request failed: {exc!r}") from exc
        else:
            if resp.status_code == 200:
                return resp.json()
            retryable = resp.status_code == 429 or resp.status_code >= 500
            if not retryable or attempt == retries:
                request_id = resp.headers.get("request-id") or resp.headers.get("x-request-id")
                raise JevError(f"Jev HTTP {resp.status_code} (request id {request_id}): {resp.text[:300]}")
        time.sleep(0.5 * 2**attempt)
    raise JevError("unreachable")
