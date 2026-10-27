"""
Latent-space utilities: how a seed becomes a latent vector, and how two
latent vectors are blended (interpolated).

Everything here is pure NumPy so it can be unit-tested without a model.

Key ideas
---------
* A StyleGAN2 latent code ``z`` is a 512-dim vector sampled from a standard
  normal distribution N(0, I).  A *seed* makes that sampling reproducible:
  the same seed always gives the same ``z`` and therefore the same image.
* Interpolation walks from ``z_A`` (t = 0) to ``z_B`` (t = 1).
    - LERP  : straight line  z(t) = (1 - t) z_A + t z_B
    - SLERP : walks along the great circle between the two vectors.  In high
              dimensions Gaussian samples live near a sphere of radius
              sqrt(dim); LERP cuts through the "hollow" middle (norm dips),
              SLERP stays on the shell, so its midpoints look more "typical".
* In W-space mode the endpoints are first passed through the mapping
  network (z -> w) and the *w* vectors are interpolated linearly, which is
  usually the smoothest because W is less entangled than Z.
"""

from __future__ import annotations

import numpy as np

INTERPOLATION_METHODS = ("slerp", "lerp", "w_lerp")


# --------------------------------------------------------------------------
# Latent representations
# --------------------------------------------------------------------------
def seed_to_z(seed: int, z_dim: int = 512) -> np.ndarray:
    """Deterministically map an integer seed to a latent vector z ~ N(0, I).

    Uses the same convention as NVIDIA's ``gen_images.py``
    (``np.random.RandomState(seed).randn(1, z_dim)``) so seeds produce the
    same faces as the official StyleGAN tooling.
    """
    seed = int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in the range [0, 2**32 - 1]")
    if z_dim <= 0:
        raise ValueError("z_dim must be positive")
    return np.random.RandomState(seed).randn(z_dim).astype(np.float32)


def latent_stats(z: np.ndarray) -> dict:
    """Summary statistics used by the UI and by the tests."""
    z = np.asarray(z, dtype=np.float64)
    return {
        "dim": int(z.size),
        "mean": float(z.mean()),
        "std": float(z.std()),
        "norm": float(np.linalg.norm(z)),
        "expected_norm": float(np.sqrt(z.size)),
        "min": float(z.min()),
        "max": float(z.max()),
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


# --------------------------------------------------------------------------
# Interpolation
# --------------------------------------------------------------------------
def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation: (1 - t) * a + t * b."""
    _check_t(t)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    _check_shapes(a, b)
    return ((1.0 - t) * a + t * b).astype(np.float32)


def slerp(a: np.ndarray, b: np.ndarray, t: float, eps: float = 1e-7) -> np.ndarray:
    """Spherical linear interpolation.

    The angle between the vectors is interpolated and the vector length is
    blended linearly, so the result stays close to the Gaussian "shell".
    Falls back to LERP when the vectors are (almost) parallel.
    """
    _check_t(t)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    _check_shapes(a, b)

    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return lerp(a, b, t)
    ua, ub = a / na, b / nb
    dot = float(np.clip(ua @ ub, -1.0, 1.0))
    omega = np.arccos(dot)
    if np.sin(omega) < eps:  # parallel / anti-parallel -> no unique great circle
        return lerp(a, b, t)
    direction = (np.sin((1 - t) * omega) * ua + np.sin(t * omega) * ub) / np.sin(omega)
    length = (1 - t) * na + t * nb
    return (direction * length).astype(np.float32)


def interpolate(a: np.ndarray, b: np.ndarray, t: float, method: str = "slerp") -> np.ndarray:
    """Dispatch helper. ``w_lerp`` is plain LERP applied to W vectors."""
    if method == "slerp":
        return slerp(a, b, t)
    if method in ("lerp", "w_lerp"):
        return lerp(a, b, t)
    raise ValueError(f"unknown interpolation method {method!r}; use one of {INTERPOLATION_METHODS}")


def interpolation_ts(steps: int) -> np.ndarray:
    """``steps`` evenly spaced t values from 0 to 1 *inclusive* (endpoints included)."""
    steps = int(steps)
    if steps < 2:
        raise ValueError("steps must be >= 2 (the two endpoints)")
    return np.linspace(0.0, 1.0, steps, dtype=np.float64)


def interpolation_path(a: np.ndarray, b: np.ndarray, steps: int, method: str = "slerp") -> np.ndarray:
    """Return an array of shape (steps, dim) walking from ``a`` to ``b``."""
    return np.stack([interpolate(a, b, float(t), method) for t in interpolation_ts(steps)])


# --------------------------------------------------------------------------
def _check_t(t: float) -> None:
    if not 0.0 <= float(t) <= 1.0:
        raise ValueError("t must be in [0, 1]")


def _check_shapes(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
