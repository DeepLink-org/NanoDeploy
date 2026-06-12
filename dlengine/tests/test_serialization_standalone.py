"""Standalone serialization sanity test for the current dlengine C++ binding."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _skip(msg: str) -> int:
    if "pytest" in sys.modules:
        import pytest

        pytest.skip(msg)
    print(f"[skip] {msg}")
    return 0


def _import_cpp():
    try:
        import dlengine._cpp as cpp

        return cpp
    except Exception as e:
        return _skip(f"dlengine._cpp not importable: {type(e).__name__}: {e}")


def test_serialization():
    cpp = _import_cpp()
    if isinstance(cpp, int):
        return

    token_ids = [101, 202, 303, 404]
    seq = cpp.Sequence(token_ids, cpp.SamplingParams(0.7, 128, False, False))
    seq.seq_id = 999
    seq.status = cpp.SequenceStatus.RUNNING

    ctx = seq.block_ctx(cpp.BlockContextSlot.ACTIVE)
    ctx.reset("test_engine", 1, 1, 16)
    ctx.block_location = [cpp.BlockLocation(0, 50)]

    buf_size = 4096
    buf = (ctypes.c_ubyte * buf_size)()
    addr = ctypes.addressof(buf)

    size = cpp.serialize(addr, buf_size, [seq], True)
    seqs_out = cpp.deserialize(addr, size)

    assert len(seqs_out) == 1
    s2 = seqs_out[0]
    assert s2.token_ids == token_ids
    assert s2.seq_id == 999
    assert s2.status == cpp.SequenceStatus.RUNNING

    ctx2 = s2.block_ctx(cpp.BlockContextSlot.ACTIVE)
    assert ctx2.engine_id == "test_engine"
    loc = list(ctx2.block_location)[0]
    assert (loc.first, loc.second) == (0, 50)

    ids = s2.token_ids
    ids.append(555)
    s2.token_ids = ids
    assert s2.token_ids[-1] == 555
    locations = list(ctx2.block_location)
    locations.append(cpp.BlockLocation(1, 60))
    ctx2.block_location = locations
    assert len(ctx2.block_location) == 2

    print("PASS: Serialization round-trip and mutability verified.")


if __name__ == "__main__":
    test_serialization()
