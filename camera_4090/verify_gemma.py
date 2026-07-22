"""One-token warmup check for the Gemma OpenAI-compatible endpoint."""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env.vlm")


def main() -> int:
    parser = argparse.ArgumentParser(description="Warm Gemma before starting a robot-planning session.")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()

    from vlm_terminal import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

    started_at = time.perf_counter()
    try:
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, timeout=args.timeout)
        client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "ready"}],
            max_tokens=1,
            temperature=0,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"GEMMA NOT READY: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "ready": True,
                "model": OPENAI_MODEL,
                "base_url": OPENAI_BASE_URL,
                "warmup_latency_ms": int((time.perf_counter() - started_at) * 1000),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
