"""Test cases: LATENT REPRESENTATIONS - how a seed becomes a latent vector z."""
import numpy as np
import pytest

from lse import latent as L


def test_seed_gives_512_dim_float_vector():
    z = L.seed_to_z(42)
    assert z.shape == (512,)
    assert z.dtype == np.float32


def test_same_seed_is_deterministic():
    np.testing.assert_array_equal(L.seed_to_z(7), L.seed_to_z(7))


def test_different_seeds_give_different_vectors():
    assert not np.allclose(L.seed_to_z(1), L.seed_to_z(2))


def test_matches_official_nvidia_convention():
    # gen_images.py uses np.random.RandomState(seed).randn(1, z_dim)
    expected = np.random.RandomState(123).randn(1, 512)[0].astype(np.float32)
    np.testing.assert_allclose(L.seed_to_z(123), expected)


def test_z_is_standard_normal():
    zs = np.stack([L.seed_to_z(s) for s in range(200)])
    assert abs(zs.mean()) < 0.02
    assert abs(zs.std() - 1.0) < 0.02


def test_norm_concentrates_near_sqrt_dim():
    # high-dimensional Gaussian samples lie close to a sphere of radius sqrt(d)
    norms = [np.linalg.norm(L.seed_to_z(s)) for s in range(100)]
    assert abs(np.mean(norms) - np.sqrt(512)) < 0.5
    assert np.std(norms) < 1.5


def test_random_latents_are_nearly_orthogonal():
    cos = [L.cosine_similarity(L.seed_to_z(s), L.seed_to_z(s + 1000)) for s in range(50)]
    assert abs(np.mean(cos)) < 0.05
    assert max(abs(c) for c in cos) < 0.25


def test_latent_stats_fields():
    st = L.latent_stats(L.seed_to_z(0))
    assert st["dim"] == 512
    assert st["expected_norm"] == pytest.approx(np.sqrt(512))
    assert st["min"] < st["mean"] < st["max"]


@pytest.mark.parametrize("bad", [-1, 2**32])
def test_invalid_seed_rejected(bad):
    with pytest.raises(ValueError):
        L.seed_to_z(bad)


def test_custom_dimension():
    assert L.seed_to_z(5, z_dim=64).shape == (64,)
