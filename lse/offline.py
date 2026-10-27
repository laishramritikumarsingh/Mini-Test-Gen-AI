"""
Build ONE self-contained HTML file that works with a double-click, with no Python
and no internet.

The real pretrained StyleGAN2 generates every frame ahead of time (dense path of
``DENSE`` frames per seed pair and interpolation method). The page embeds those
images and latent data, plus the normal frontend (HTML/CSS/JS), and a small
"offline shim" that answers the frontend's /api/* requests from the embedded data.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import latent as L
from .payload import jpeg_data_url, run_payload, snip

STATIC = Path(__file__).resolve().parent.parent / "static"
DENSE = 25                         # frames rendered per path (slider resolution)
KEYFRAMES = [0, 3, 6, 9, 12, 15, 18, 21, 24]   # shown in the filmstrip (9 frames)
DEFAULT_PAIRS = [(42, 1234), (7, 2026), (100, 555), (3, 19)]


def build_data(g, pairs, truncation_psi: float = 0.7, progress=print) -> dict:
    runs = {}
    total = len(pairs) * len(L.INTERPOLATION_METHODS)
    n = 0
    for sa, sb in pairs:
        for method in L.INTERPOLATION_METHODS:
            n += 1
            progress(f"[offline] {n}/{total}: seed {sa} -> {sb} ({method}) ...")
            t0 = time.perf_counter()
            ts, imgs, z_path, w_path = g.interpolate_images(sa, sb, DENSE, method, truncation_psi)
            p = dict(seed_a=sa, seed_b=sb, steps=len(KEYFRAMES), method=method, truncation_psi=truncation_psi)
            payload = run_payload(g, p, ts, imgs, z_path, w_path, time.perf_counter() - t0,
                                  encode=jpeg_data_url, keep=KEYFRAMES)
            # dense frames for the slider (keyframes reuse their image)
            key_img = {i: payload["frames"][k]["image"] for k, i in enumerate(KEYFRAMES)}
            payload["dense"] = [{
                "t": round(float(ts[i]), 4),
                "image": key_img.get(i) or jpeg_data_url(imgs[i]),
                "z_snippet": snip(z_path[i]) if z_path is not None else None,
                "w_snippet": snip(w_path[i, 0]),
                "z_norm": round(float(np.linalg.norm(z_path[i])), 3) if z_path is not None else None,
            } for i in range(len(ts))]
            runs[f"{sa}-{sb}-{method}"] = payload
    info = g.info()
    return {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "model": info,
        "pairs": [list(p) for p in pairs],
        "truncation_psi": truncation_psi,
        "runs": runs,
    }


def build_page(g, pairs=None, truncation_psi: float = 0.7, progress=print) -> str:
    data = build_data(g, pairs or DEFAULT_PAIRS, truncation_psi, progress)
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    charts = (STATIC / "charts.js").read_text(encoding="utf-8")
    shim = (STATIC / "offline_shim.js").read_text(encoding="utf-8")
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")

    html = html.replace('<link rel="stylesheet" href="/static/style.css" />', f"<style>\n{css}\n</style>")
    html = html.replace('<script src="/static/charts.js"></script>', f"<script>\n{charts}\n</script>")
    html = html.replace(
        '<script src="/static/app.js"></script>',
        f'<script>window.LSE_OFFLINE = {blob};</script>\n<script>\n{shim}\n</script>\n<script>\n{app}\n</script>',
    )
    html = html.replace("<title>Latent Space Explorer</title>", "<title>Latent Space Explorer (offline)</title>")
    return html
