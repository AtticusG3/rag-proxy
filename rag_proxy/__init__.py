"""Cognitive RAG proxy package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from rag_proxy.app import app

        return app
    raise AttributeError(name)
