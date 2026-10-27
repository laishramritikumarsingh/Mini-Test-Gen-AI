"""
Latent Space Explorer - Flask backend.

Run:  python app.py              (downloads the default model on first run, then opens the browser)
      python app.py --help       (all options)
"""

from __future__ import annotations

import argparse
import os
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
import io

from lse import exporter, offline, payload
from lse import latent as L
from lse.generator import (DEFAULT_MODEL, MODELS, FrameCache, LatentGenerator, download_weights,
                           find_local_checkpoints, to_data_url, weights_path)

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
CORS(app, resources={r"/api/*": {"origins": ["https://dynamic-cannoli-b98e9f.netlify.app"]}})

MAX_STEPS = 24

STATE = {
    "status": "idle",          # idle | downloading | loading | ready | error
    "message": "",
    "model_key": None,
    "gen": None,
    "last_run": None,
}
CACHE = FrameCache()
_load_lock = threading.Lock()


# --------------------------------------------------------------------------
# Model loading (background thread so the page can show progress)
# --------------------------------------------------------------------------
def _device() -> str:
    want = os.environ.get("LSE_DEVICE", "cpu").lower()
    if want == "cuda":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def load_model(key: str | None = None, checkpoint: str | None = None, untrained: bool = False) -> None:
    with _load_lock:
        try:
            if untrained:
                STATE.update(status="loading", message="Building untrained demo network ...")
                gen = LatentGenerator.random_init(resolution=64)
                STATE.update(gen=gen, model_key="untrained", status="ready",
                             message="UNTRAINED demo network (random weights) - for testing only")
                return
            if checkpoint:
                path = Path(checkpoint)
                key = path.stem
            else:
                key = key or DEFAULT_MODEL
                path = weights_path(key)
                if not path.exists():
                    STATE.update(status="downloading", message=f"Downloading {MODELS[key].filename} (first run only) ...")
                    download_weights(key)
            STATE.update(status="loading", message=f"Loading {path.name} ...")
            gen = LatentGenerator.from_pickle(path, device=_device(), name=key)
            # warm-up so the first click is fast
            gen.seed_to_image(0)
            CACHE.__init__()
            STATE.update(gen=gen, model_key=key, status="ready", message=f"Loaded {path.name}")
        except Exception as e:  # surface every failure to the UI
            STATE.update(status="error", message=f"{type(e).__name__}: {e}")
            print("[model] ERROR:", STATE["message"])


def _gen() -> LatentGenerator:
    g = STATE["gen"]
    if g is None or STATE["status"] != "ready":
        raise RuntimeError("Model is not ready yet: " + (STATE["message"] or STATE["status"]))
    return g


def _params(body: dict) -> dict:
    p = {
        "seed_a": int(body.get("seed_a", 42)),
        "seed_b": int(body.get("seed_b", 1234)),
        "steps": int(body.get("steps", 8)),
        "method": str(body.get("method", "slerp")),
        "truncation_psi": float(body.get("truncation_psi", 0.7)),
    }
    for s in ("seed_a", "seed_b"):
        if not 0 <= p[s] < 2**32:
            raise ValueError("Seeds must be between 0 and 4294967295")
    if not 2 <= p["steps"] <= MAX_STEPS:
        raise ValueError(f"Steps must be between 2 and {MAX_STEPS}")
    if p["method"] not in L.INTERPOLATION_METHODS:
        raise ValueError("Unknown interpolation method")
    if not 0.0 <= p["truncation_psi"] <= 1.5:
        raise ValueError("Truncation psi must be between 0 and 1.5")
    return p


@app.errorhandler(Exception)
def _err(e):
    code = getattr(e, "code", 400)
    return jsonify(error=str(e)), code if isinstance(code, int) else 400


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    g = STATE["gen"]
    return jsonify(
        status=STATE["status"], message=STATE["message"], model_key=STATE["model_key"],
        model=g.info() if g is not None and STATE["status"] == "ready" else None,
        models=[{"key": k, "description": m.description, "cached": weights_path(k).exists()} for k, m in MODELS.items()],
        local_checkpoints=[p.name for p in find_local_checkpoints()],
        methods=list(L.INTERPOLATION_METHODS), max_steps=MAX_STEPS,
    )


@app.post("/api/load_model")
def api_load_model():
    key = (request.get_json(force=True) or {}).get("key", DEFAULT_MODEL)
    if key not in MODELS:
        raise ValueError("Unknown model")
    if STATE["status"] in ("downloading", "loading"):
        raise ValueError("A model is already loading - please wait")
    STATE.update(status="loading", message="Starting ...")
    threading.Thread(target=load_model, kwargs={"key": key}, daemon=True).start()
    return jsonify(ok=True)


@app.post("/api/interpolate")
def api_interpolate():
    g = _gen()
    p = _params(request.get_json(force=True) or {})
    t0 = time.perf_counter()
    ts, imgs, z_path, w_path = g.interpolate_images(p["seed_a"], p["seed_b"], p["steps"], p["method"], p["truncation_psi"])
    elapsed = time.perf_counter() - t0

    STATE["last_run"] = {"params": p, "ts": ts, "images": imgs, "z_path": z_path, "w_path": w_path}
    return jsonify(payload.run_payload(g, p, ts, imgs, z_path, w_path, elapsed))


@app.post("/api/frame")
def api_frame():
    """Render the image at an arbitrary slider position t (continuous)."""
    g = _gen()
    body = request.get_json(force=True) or {}
    p = _params(body)
    t = float(body.get("t", 0.5))
    if not 0 <= t <= 1:
        raise ValueError("t must be in [0, 1]")
    t = round(t, 3)
    key = (STATE["model_key"], p["seed_a"], p["seed_b"], p["method"], p["truncation_psi"], t)
    hit = CACHE.get(key)
    if hit is None:
        zt, wt = g.latent_at(p["seed_a"], p["seed_b"], t, p["method"], p["truncation_psi"])
        img = g.w_to_image(wt)[0]
        hit = {
            "t": t, "image": to_data_url(img),
            "z_snippet": payload.snip(zt) if zt is not None else None,
            "w_snippet": payload.snip(wt[0]),
            "z_norm": round(float(np.linalg.norm(zt)), 3) if zt is not None else None,
        }
        CACHE.put(key, hit)
    return jsonify(hit)


@app.post("/api/export")
def api_export():
    run = STATE["last_run"]
    if run is None:
        raise ValueError("Generate an interpolation first")
    files = exporter.build_files(_gen().info(), run["params"], run["ts"], run["images"], run["z_path"], run["w_path"])
    name = exporter.run_name(run["params"])
    out = exporter.save_run(files, name)
    print(f"[export] saved {len(files)} files to {out}")
    return send_file(io.BytesIO(exporter.zip_bytes(files, name)), mimetype="application/zip",
                     as_attachment=True, download_name=f"{name}.zip")


@app.post("/api/offline_page")
def api_offline_page():
    """Build a single self-contained HTML file for the current seeds (opens by double-click, no Python)."""
    g = _gen()
    p = _params(request.get_json(force=True) or {})
    html = offline.build_page(g, [(p["seed_a"], p["seed_b"])], truncation_psi=p["truncation_psi"])
    name = f"Latent_Space_Explorer_OFFLINE_seed{p['seed_a']}-{p['seed_b']}.html"
    (exporter.OUTPUTS_DIR / name).write_text(html, encoding="utf-8")
    return send_file(io.BytesIO(html.encode("utf-8")), mimetype="text/html", as_attachment=True, download_name=name)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Latent Space Explorer (StyleGAN2)")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS), help="pretrained model to use")
    ap.add_argument("--checkpoint", help="path to any StyleGAN2 .pkl (overrides --model)")
    ap.add_argument("--port", type=int, default=8765)  # (5000 clashes with AirPlay on macOS)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    ap.add_argument("--untrained-demo", action="store_true",
                    help="use a random, UNTRAINED StyleGAN2 (no download) - only to check the app works")
    args = ap.parse_args()

    threading.Thread(target=load_model, kwargs=dict(key=args.model, checkpoint=args.checkpoint,
                                                    untrained=args.untrained_demo), daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    print(f"\n  Latent Space Explorer running at  {url}\n  (press Ctrl+C to stop)\n")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
