from __future__ import annotations

import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def require_dlengine_cpp():
    return pytest.importorskip(
        "dlengine._cpp",
        reason="dlengine C++ extension is not built or installed",
    )


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test", allow_module_level=True)
