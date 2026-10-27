"""Builds the JSON data the frontend draws (shared by the live app and the offline page)."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from . import latent as L
from .generator import to_data_url

SNIPPET = 48


def snip(v) -> list:
    return [round(float(x), 3) for x in np.asarray(v).ravel()[:SNIPPET]]


def hist(z, bins=24, lo=-4.0, hi=4.0) -> list:
    h, _ = np.histogram(z, bins=bins, range=(lo, hi))
    return h.tolist()


def jpeg_data_url(arr: np.ndarray, quality: int = 88) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def frame_record(i, t, imgs, z_path, w_path, path_vecs, encode=to_data_url) -> dict:
    a_vec, b_vec = path_vecs[0], path_vecs[-1]
    return {
        "t": round(float(t), 4),
        "image": encode(imgs[i]),
        "z_snippet": snip(z_path[i]) if z_path is not None else None,
        "w_snippet": snip(w_path[i, 0]),
        "z_norm": round(float(np.linalg.norm(z_path[i])), 3) if z_path is not None else None,
        "w_norm": round(float(np.linalg.norm(w_path[i, 0])), 3),
        "cos_to_a": round(L.cosine_similarity(path_vecs[i], a_vec), 4),
        "cos_to_b": round(L.cosine_similarity(path_vecs[i], b_vec), 4),
        "pixel_change": round(float(np.abs(imgs[i].astype(np.float32) - imgs[i - 1]).mean()), 3) if i else 0.0,
        "pixel_dist_from_a": round(float(np.abs(imgs[i].astype(np.float32) - imgs[0]).mean()), 3),
    }


def run_payload(g, p: dict, ts, imgs, z_path, w_path, elapsed: float, encode=to_data_url, keep=None) -> dict:
    """Response for /api/interpolate. ``keep`` = indices of frames to include (default all)."""
    za, zb = L.seed_to_z(p["seed_a"], g.z_dim), L.seed_to_z(p["seed_b"], g.z_dim)
    path_vecs = z_path if z_path is not None else w_path[:, 0, :]
    idx = list(range(len(ts))) if keep is None else list(keep)
    frames = []
    for k, i in enumerate(idx):
        fr = frame_record(i, ts[i], imgs, z_path, w_path, path_vecs, encode)
        # pixel change relative to the previous *kept* frame
        if k:
            fr["pixel_change"] = round(float(np.abs(imgs[i].astype(np.float32) - imgs[idx[k - 1]]).mean()), 3)
        frames.append(fr)

    dense = np.linspace(0, 1, 41)
    norm_curve = {
        "t": dense.round(3).tolist(),
        "lerp": [round(float(np.linalg.norm(L.lerp(za, zb, t))), 3) for t in dense],
        "slerp": [round(float(np.linalg.norm(L.slerp(za, zb, t))), 3) for t in dense],
        "expected": round(float(np.sqrt(g.z_dim)), 3),
    }
    return dict(
        params=dict(p, steps=len(idx)), frames=frames, elapsed_sec=round(elapsed, 2), norm_curve=norm_curve,
        angle_deg=round(float(np.degrees(np.arccos(np.clip(L.cosine_similarity(za, zb), -1, 1)))), 2),
        endpoint_stats={"a": L.latent_stats(za), "b": L.latent_stats(zb)},
        histograms={"a": hist(za), "b": hist(zb), "range": [-4, 4]},
    )
