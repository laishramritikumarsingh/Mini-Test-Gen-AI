"""Test cases: INTERPOLATION between two latent vectors."""
import numpy as np
import pytest

from lse import latent as L

ZA, ZB = L.seed_to_z(42), L.seed_to_z(1234)


@pytest.mark.parametrize("method", ["lerp", "slerp"])
def test_endpoints_are_exact(method):
    np.testing.assert_allclose(L.interpolate(ZA, ZB, 0.0, method), ZA, atol=1e-5)
    np.testing.assert_allclose(L.interpolate(ZA, ZB, 1.0, method), ZB, atol=1e-5)


def test_lerp_midpoint_is_average():
    np.testing.assert_allclose(L.lerp(ZA, ZB, 0.5), (ZA + ZB) / 2, atol=1e-6)


def test_lerp_norm_dips_in_the_middle():
    # LERP between two ~orthogonal vectors shrinks the norm by about 1/sqrt(2)
    mid = np.linalg.norm(L.lerp(ZA, ZB, 0.5))
    ends = (np.linalg.norm(ZA) + np.linalg.norm(ZB)) / 2
    assert mid < 0.8 * ends


def test_slerp_preserves_norm():
    na, nb = np.linalg.norm(ZA), np.linalg.norm(ZB)
    for t in np.linspace(0, 1, 11):
        n = np.linalg.norm(L.slerp(ZA, ZB, t))
        assert n == pytest.approx((1 - t) * na + t * nb, rel=1e-4)


def test_slerp_angle_moves_uniformly():
    total = np.arccos(L.cosine_similarity(ZA, ZB))
    for t in (0.25, 0.5, 0.75):
        ang = np.arccos(np.clip(L.cosine_similarity(ZA, L.slerp(ZA, ZB, t)), -1, 1))
        assert ang == pytest.approx(t * total, rel=1e-3)


def test_path_shape_and_order():
    path = L.interpolation_path(ZA, ZB, 8, "slerp")
    assert path.shape == (8, 512)
    sim_to_b = [L.cosine_similarity(p, ZB) for p in path]
    assert all(np.diff(sim_to_b) > 0)  # moves monotonically towards B


def test_ts_include_endpoints_and_are_even():
    ts = L.interpolation_ts(5)
    np.testing.assert_allclose(ts, [0, 0.25, 0.5, 0.75, 1])


def test_slerp_parallel_vectors_fall_back_to_lerp():
    np.testing.assert_allclose(L.slerp(ZA, 2 * ZA, 0.5), 1.5 * ZA, rtol=1e-5)


def test_identical_vectors_constant_path():
    path = L.interpolation_path(ZA, ZA, 4, "slerp")
    for p in path:
        np.testing.assert_allclose(p, ZA, atol=1e-5)


@pytest.mark.parametrize("kwargs, exc", [
    (dict(t=1.5), ValueError), (dict(t=-0.1), ValueError), (dict(t=0.5, method="cubic"), ValueError),
])
def test_invalid_arguments(kwargs, exc):
    with pytest.raises(exc):
        L.interpolate(ZA, ZB, **kwargs)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        L.lerp(ZA, ZA[:10], 0.5)


def test_steps_must_include_two_endpoints():
    with pytest.raises(ValueError):
        L.interpolation_ts(1)
