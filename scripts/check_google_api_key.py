#!/usr/bin/env python3
"""
Quick checker for Google Generative Language API keys.

Usage:
  python scripts/check_google_api_key.py --api-key YOUR_KEY
  GOOGLE_API_KEY=YOUR_KEY python scripts/check_google_api_key.py
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


def check_api_key(api_key: str, timeout: float = 10.0) -> int:
    # A lightweight endpoint that validates key/auth and API enablement.
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    req = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body) if body else {}
            model_count = len(data.get("models", []))
            print(f"OK: API key works (HTTP {resp.status}).")
            print(f"Visible models: {model_count}")
            return 0

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        message = body
        try:
            parsed = json.loads(body)
            message = (
                parsed.get("error", {}).get("message")
                or parsed.get("message")
                or body
            )
        except json.JSONDecodeError:
            pass

        print(f"FAIL: HTTP {e.code}")
        print(message.strip())

        if e.code == 400:
            print("Likely causes: malformed key, wrong endpoint, or disabled API.")
        elif e.code == 403:
            print("Likely causes: invalid key, key restrictions, or no API access.")
        elif e.code == 429:
            print("Rate limit/quota exceeded for this key/project.")
        return 1

    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}")
        return 2
def main() -> int:
    parser = argparse.ArgumentParser(description="Check if a Google API key works.")
    parser.add_argument("--api-key", help="Google API key (or set GOOGLE_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Provide --api-key or set GOOGLE_API_KEY.")
        return 2

    return check_api_key(api_key.strip())


if __name__ == "__main__":
    raise SystemExit(main())
