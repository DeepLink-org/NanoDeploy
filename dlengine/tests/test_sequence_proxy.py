"""Sanity checks for C++ Sequence Python bindings.

Run with:
  python tests/test_sequence_proxy.py

It exits 0 on success, or when the C++ extension is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _skip(msg: str) -> int:
    print(f"[skip] {msg}")
    return 0


def main() -> int:
    try:
        from dlengine._cpp import (
            BlockContextSlot,
            BlockLocation,
            SamplingParams,
            Sequence,
        )
    except Exception as e:
        return _skip(f"dlengine._cpp not importable: {type(e).__name__}: {e}")

    seq = Sequence([1, 2, 3], SamplingParams(1.0, 16, False, False))
    seq.active("engine", 1, 1, 16)

    ctx = seq.block_ctx(BlockContextSlot.ACTIVE)

    ctx.block_location = [BlockLocation(0, 42)]
    assert len(ctx.block_location) == 1
    loc = ctx.block_location[0]
    assert (loc.first, loc.second) == (0, 42)

    ctx.dp_idx = 123
    assert seq.block_ctx(BlockContextSlot.ACTIVE).dp_idx == 123

    print("[ok] C++ Sequence bindings preserve assigned block context state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
