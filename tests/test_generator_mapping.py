"""Test cases: HOW THE GENERATOR MAPS LATENT VECTORS TO IMAGES.

Most tests use a tiny randomly-initialised StyleGAN2 (same architecture,
untrained) so they run offline in seconds. The last group runs against the
real pretrained checkpoint when one is cached in weights/.
"""
import numpy as np
import pytest

from lse import exporter
from lse import latent as L


# ---------- z -> w (mapping network) ----------
def test_mapping_output_shape(tiny_gen):
    w = tiny_gen.z_to_w(L.seed_to_z(1))
    assert w.shape == (1, tiny_gen.num_ws, tiny_gen.w_dim)


def test_mapping_broadcasts_same_w_to_every_layer(tiny_gen):
    w = tiny_gen.z_to_w(L.seed_to_z(1))[0]
    np.testing.assert_allclose(w, np.tile(w[0], (tiny_gen.num_ws, 1)))


def test_truncation_zero_collapses_to_average(tiny_gen):
    w1 = tiny_gen.z_to_w(L.seed_to_z(1), truncation_psi=0.0)
    w2 = tiny_gen.z_to_w(L.seed_to_z(2), truncation_psi=0.0)
    np.testing.assert_allclose(w1, w2, atol=1e-5)  # every z -> w_avg


def test_truncation_pulls_towards_average(tiny_gen):
    z = L.seed_to_z(3)
    w_avg = tiny_gen.z_to_w(z, 0.0)[0, 0]
    d_half = np.linalg.norm(tiny_gen.z_to_w(z, 0.5)[0, 0] - w_avg)
    d_full = np.linalg.norm(tiny_gen.z_to_w(z, 1.0)[0, 0] - w_avg)
    assert d_half == pytest.approx(0.5 * d_full, rel=1e-4)


def test_wrong_z_dimension_rejected(tiny_gen):
    with pytest.raises(ValueError):
        tiny_gen.z_to_w(np.zeros(10, dtype=np.float32))


# ---------- w -> image (synthesis network) ----------
def test_image_shape_dtype_range(tiny_gen):
    img = tiny_gen.seed_to_image(5)
    r = tiny_gen.resolution
    assert img.shape == (r, r, 3)
    assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255


def test_generation_is_deterministic(tiny_gen):
    np.testing.assert_array_equal(tiny_gen.seed_to_image(9), tiny_gen.seed_to_image(9))


def test_different_latents_give_different_images(tiny_gen):
    assert not np.array_equal(tiny_gen.seed_to_image(1), tiny_gen.seed_to_image(2))


def test_batch_matches_single(tiny_gen):
    zs = np.stack([L.seed_to_z(s) for s in (1, 2, 3)])
    batch = tiny_gen.z_to_image(zs)
    for i, s in enumerate((1, 2, 3)):
        np.testing.assert_allclose(batch[i].astype(int), tiny_gen.seed_to_image(s).astype(int), atol=1)


def test_small_latent_change_gives_small_image_change(tiny_gen):
    z = L.seed_to_z(11)
    near = tiny_gen.z_to_image(z + 0.01 * L.seed_to_z(99))[0].astype(float)
    far = tiny_gen.z_to_image(L.seed_to_z(12))[0].astype(float)
    base = tiny_gen.z_to_image(z)[0].astype(float)
    assert np.abs(near - base).mean() < np.abs(far - base).mean()


# ---------- full interpolation pipeline ----------
@pytest.mark.parametrize("method", L.INTERPOLATION_METHODS)
def test_interpolation_endpoints_match_seed_images(tiny_gen, method):
    ts, imgs, z_path, w_path = tiny_gen.interpolate_images(4, 8, steps=6, method=method)
    assert len(ts) == len(imgs) == 6
    assert w_path.shape == (6, tiny_gen.num_ws, tiny_gen.w_dim)
    np.testing.assert_allclose(imgs[0].astype(int), tiny_gen.seed_to_image(4).astype(int), atol=1)
    np.testing.assert_allclose(imgs[-1].astype(int), tiny_gen.seed_to_image(8).astype(int), atol=1)
    assert (z_path is None) == (method == "w_lerp")


def test_intermediate_images_are_between_endpoints(tiny_gen):
    _, imgs, _, _ = tiny_gen.interpolate_images(4, 8, steps=8, method="w_lerp")
    d = [np.abs(im.astype(float) - imgs[0]).mean() for im in imgs]
    assert d[0] == 0 and d[-1] > 0
    assert d[len(d) // 2] < d[-1] * 1.5  # mid frame is not further than the far endpoint


def test_latent_at_matches_path(tiny_gen):
    _, imgs, _, _ = tiny_gen.interpolate_images(1, 2, steps=5, method="slerp")
    _, w = tiny_gen.latent_at(1, 2, 0.5, "slerp", 0.7)
    np.testing.assert_allclose(tiny_gen.w_to_image(w)[0].astype(int), imgs[2].astype(int), atol=1)


def test_export_contains_endpoints_and_intermediates(tiny_gen):
    params = dict(seed_a=1, seed_b=2, steps=5, method="slerp", truncation_psi=0.7)
    ts, imgs, zp, wp = tiny_gen.interpolate_images(1, 2, 5, "slerp")
    files = exporter.build_files(tiny_gen.info(), params, ts, imgs, zp, wp)
    assert "endpoint_A_seed1.png" in files and "endpoint_B_seed2.png" in files
    assert len([f for f in files if f.startswith("intermediate/")]) == 3
    assert len([f for f in files if f.startswith("all_frames/")]) == 5
    for f in ("interpolation_grid.png", "interpolation.gif", "latents.npz", "metadata.json"):
        assert f in files


# ---------- the real pretrained model (skipped if not downloaded) ----------
def test_pretrained_model_loads_and_generates(pretrained_gen):
    assert pretrained_gen.pretrained
    assert pretrained_gen.z_dim == 512
    img = pretrained_gen.seed_to_image(42)
    assert img.shape == (pretrained_gen.resolution, pretrained_gen.resolution, 3)
    assert img.std() > 10  # a real picture, not a flat colour


def test_pretrained_interpolation(pretrained_gen):
    ts, imgs, _, _ = pretrained_gen.interpolate_images(42, 1234, steps=4, method="slerp")
    assert len(imgs) == 4
    changes = [np.abs(imgs[i].astype(float) - imgs[i - 1]).mean() for i in range(1, 4)]
    assert all(c > 0 for c in changes)


# ---------- offline single-file page ----------
def test_offline_page_is_self_contained(tiny_gen):
    from lse import offline
    html = offline.build_page(tiny_gen, [(1, 2)], progress=lambda *_: None)
    assert "window.LSE_OFFLINE" in html
    assert "/static/" not in html          # nothing loaded from a server
    assert "http://" not in html.split("window.LSE_OFFLINE")[0].replace("http://www.w3.org", "")
    assert html.count("data:image/jpeg") >= offline.DENSE  # every slider frame embedded
