#!/usr/bin/env python3
"""Quick post-deploy smoke checks on buster (run on host)."""
from __future__ import annotations

import sys

import httpx


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"[ok] {name}")
    except Exception as exc:
        print(f"[X] {name}: {exc}")
        sys.exit(1)


def main() -> None:
    check(
        "embed :18089",
        lambda: httpx.post(
            "http://127.0.0.1:18089/v1/embeddings",
            json={"model": "nomic-embed-text-v1.5", "input": "deploy-check"},
            timeout=30.0,
        ).raise_for_status(),
    )
    check(
        "proxy metrics :8081",
        lambda: httpx.get("http://127.0.0.1:8081/metrics", timeout=10.0).raise_for_status(),
    )
    check(
        "admin health :8087",
        lambda: httpx.get("http://127.0.0.1:8087/health", timeout=10.0).raise_for_status(),
    )
    print("[done] all smoke checks passed")


if __name__ == "__main__":
    main()
