"""Small, dependency-free helpers for reporting effective DiT FP8 state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_STREAMS = ("img", "txt")


def fp8_precision_status(
    transformer: Any,
    requested_by_stream: Mapping[str, bool],
) -> dict[str, dict[str, Any]]:
    """Return requested and installed FP8 state for each DiT stream.

    FP8 weights are installed lazily during block forwards, so a requested
    stream with no converted blocks is pending rather than bf16.
    """
    blocks = tuple(getattr(transformer, "double_blocks", ()))
    total_blocks = len(blocks)
    result: dict[str, dict[str, Any]] = {}

    for stream in _STREAMS:
        requested = bool(requested_by_stream.get(stream, False))
        installed_blocks = sum(
            bool(getattr(block, f"_fp8_{stream}_installed", False))
            for block in blocks
        )
        if total_blocks == 0:
            effective = "unknown"
        elif installed_blocks == total_blocks:
            effective = "fp8"
        elif installed_blocks:
            effective = "mixed"
        elif requested:
            effective = "pending"
        else:
            effective = "bf16"
        result[stream] = {
            "requested": requested,
            "effective": effective,
            "installed_blocks": installed_blocks,
            "total_blocks": total_blocks,
        }

    return result


def format_fp8_precision_status(status: Mapping[str, Mapping[str, Any]]) -> str:
    """Format per-stream status for a concise runtime log line."""
    parts = []
    for stream in _STREAMS:
        item = status[stream]
        effective = item["effective"]
        requested = "yes" if item["requested"] else "no"
        installed = item["installed_blocks"]
        total = item["total_blocks"]
        if effective == "fp8":
            detail = f"requested={requested}; {installed}/{total} blocks converted"
        elif effective == "bf16":
            detail = f"requested={requested}; FP8 not requested"
        elif effective == "pending":
            detail = f"requested={requested}; {installed}/{total} blocks converted lazily"
        elif effective == "mixed":
            detail = (
                f"requested={requested}; FP8 active in {installed}/{total} blocks; "
                "conversion incomplete"
            )
        else:
            detail = "no DiT blocks found"
        parts.append(f"{stream}={effective} ({detail})")
    return "DiT precision: " + ", ".join(parts)
