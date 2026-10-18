import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def public_set():
    def _public_set(n: int) -> Path:
        return REPO_ROOT / "testcase" / "2026_08_20_samples" / "problem" / f"benchmark_set_{n}"

    return _public_set
