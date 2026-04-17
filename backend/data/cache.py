"""Tiny disk cache for expensive pybaseball / ESPN pulls."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable

from backend.config import CONFIG


def _key(name: str, args: tuple, kwargs: dict) -> Path:
    h = hashlib.sha1(pickle.dumps((name, args, sorted(kwargs.items())))).hexdigest()[:16]
    return CONFIG.data_cache_dir / f"{name}_{h}.pkl"


def cached(name: str):
    """Decorator: memoize a pure-ish function's output to disk."""
    def deco(fn: Callable[..., Any]):
        def wrapper(*args, **kwargs):
            path = _key(name, args, kwargs)
            if path.exists():
                try:
                    with open(path, "rb") as fh:
                        return pickle.load(fh)
                except Exception:
                    path.unlink(missing_ok=True)
            result = fn(*args, **kwargs)
            try:
                with open(path, "wb") as fh:
                    pickle.dump(result, fh)
            except Exception:
                pass
            return result
        wrapper.__wrapped__ = fn
        return wrapper
    return deco
