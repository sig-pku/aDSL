from __future__ import annotations

from typing import Iterable


def set_link_color(asset, color: Iterable[float] = (1, 1, 1), *, alpha: float = 1.0):
    return asset.set_link_color(color, alpha=alpha)


def get_link_rgba(asset):
    return asset.get_link_rgba()


__all__ = ["set_link_color", "get_link_rgba"]
