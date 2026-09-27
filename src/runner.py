"""
runner.py — Run each adversarial test case against the target system prompt.

Each test is an independent conversation — no shared context between tests.
Uses claude-haiku-4-5 for the target (faster, cheaper, surfaces vulnerabilities).
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import anthropic

from attacker import AttackCase


def run_tests(
    system_prompt: str,
    cases: list[AttackCase],
    model: str = "claude-haiku-4-5-20251001",
    api_key: Optional[str] = None,
    delay: float = 0.3,
    on_progress: Optional[Callable[[int, int, AttackCase], None]] = None,
) -> list[AttackCase]:
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    total = len(cases)

    for i, case in enumerate(cases):
        if on_progress:
            on_progress(i, total, case)
        case.response = _send_with_retry(client, model, system_prompt, case.input)
        time.sleep(delay)

    if on_progress:
        on_progress(total, total, cases[-1] if cases else None)

    return cases


def _send_with_retry(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    user_input: str,
    retries: int = 1,
) -> str:
    for attempt in range(retries + 1):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=800,
                system=system_prompt,
                messages=[{"role": "user", "content": user_input}],
            )
            return "\n".join(
                block.text for block in message.content if hasattr(block, "text")
            ).strip()
        except anthropic.RateLimitError:
            if attempt < retries:
                time.sleep(5)
                continue
            return "[ERROR: Rate limit exceeded]"
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < retries:
                time.sleep(3)
                continue
            return f"[ERROR: API error {e.status_code}]"
        except Exception as e:
            return f"[ERROR: {str(e)[:100]}]"

    return "[ERROR: All retries exhausted]"
