import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lse.generator import LatentGenerator, find_local_checkpoints  # noqa: E402


@pytest.fixture(scope="session")
def tiny_gen():
    """A small, randomly initialised StyleGAN2 (real architecture, untrained weights).
    Lets the mapping tests run offline in seconds."""
    return LatentGenerator.random_init(resolution=32, seed=0)


@pytest.fixture(scope="session")
def pretrained_gen():
    """The real pretrained model, if a .pkl is cached in weights/ (otherwise skipped)."""
    if os.environ.get("LSE_SKIP_PRETRAINED"):
        pytest.skip("LSE_SKIP_PRETRAINED set")
    ckpts = find_local_checkpoints()
    if not ckpts:
        pytest.skip("no pretrained .pkl in weights/ - run download_model.py first")
    return LatentGenerator.from_pickle(ckpts[0])
