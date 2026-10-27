"""
Output writer: saves the two endpoint images, every intermediate image,
a labelled contact sheet, an animated GIF, the latent vectors and a JSON
summary into ``outputs/<run-name>/``.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import latent as L

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(images: np.ndarray, ts: np.ndarray, seed_a: int, seed_b: int, method: str,
                  thumb: int = 192, cols: int | None = None) -> Image.Image:
    """Grid of all frames with t labels; endpoints outlined."""
    n = len(images)
    cols = cols or min(n, 6)
    rows = (n + cols - 1) // cols
    pad, label_h, head_h = 12, 26, 44
    W = cols * thumb + (cols + 1) * pad
    H = head_h + rows * (thumb + label_h + pad) + pad
    sheet = Image.new("RGB", (W, H), (248, 248, 246))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 12), f"Latent interpolation  seed {seed_a} -> seed {seed_b}   ({method}, {n} frames)",
           fill=(30, 30, 30), font=_font(18))
    for i, (img, t) in enumerate(zip(images, ts)):
        r, c = divmod(i, cols)
        x = pad + c * (thumb + pad)
        y = head_h + r * (thumb + label_h + pad)
        sheet.paste(Image.fromarray(img).resize((thumb, thumb), Image.LANCZOS), (x, y))
        is_end = i in (0, n - 1)
        if is_end:
            d.rectangle([x - 3, y - 3, x + thumb + 2, y + thumb + 2], outline=(37, 99, 235), width=3)
        tag = ("A  " if i == 0 else "B  " if i == n - 1 else "") + f"t = {t:.3f}"
        d.text((x, y + thumb + 5), tag, fill=(37, 99, 235) if is_end else (90, 90, 90), font=_font(14))
    return sheet


def gif_bytes(images: np.ndarray, frame_ms: int = 120) -> bytes:
    frames = [Image.fromarray(im) for im in images]
    frames = frames + frames[-2:0:-1]  # ping-pong A -> B -> A
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=frame_ms, loop=0)
    return buf.getvalue()


def build_files(gen_info: dict, params: dict, ts, images, z_path, w_path) -> dict[str, bytes]:
    """Return {relative_path: bytes} for every output file."""
    n = len(images)
    sa, sb, method = params["seed_a"], params["seed_b"], params["method"]
    files: dict[str, bytes] = {}

    def png(arr):
        b = io.BytesIO()
        Image.fromarray(arr).save(b, format="PNG")
        return b.getvalue()

    files[f"endpoint_A_seed{sa}.png"] = png(images[0])
    files[f"endpoint_B_seed{sb}.png"] = png(images[-1])
    for i in range(1, n - 1):
        files[f"intermediate/frame_{i:02d}_t{ts[i]:.3f}.png"] = png(images[i])
    for i in range(n):
        files[f"all_frames/frame_{i:02d}_t{ts[i]:.3f}.png"] = png(images[i])

    b = io.BytesIO()
    contact_sheet(images, ts, sa, sb, method).save(b, format="PNG")
    files["interpolation_grid.png"] = b.getvalue()
    files["interpolation.gif"] = gif_bytes(images)

    b = io.BytesIO()
    arrays = {"t": np.asarray(ts), "w_path": np.asarray(w_path)}
    if z_path is not None:
        arrays["z_path"] = np.asarray(z_path)
    np.savez_compressed(b, **arrays)
    files["latents.npz"] = b.getvalue()

    frame_rows = []
    for i in range(n):
        row = {"index": i, "t": round(float(ts[i]), 4),
               "role": "endpoint A" if i == 0 else "endpoint B" if i == n - 1 else "intermediate"}
        if z_path is not None:
            row["z_norm"] = round(float(np.linalg.norm(z_path[i])), 4)
        if i > 0:
            row["pixel_change_from_prev"] = round(float(np.abs(images[i].astype(np.float32) - images[i - 1]).mean()), 3)
        frame_rows.append(row)

    meta = {
        "task": "Latent Space Explorer - StyleGAN2 latent interpolation",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": gen_info,
        "parameters": params,
        "endpoint_A": {"seed": sa, **(L.latent_stats(z_path[0]) if z_path is not None else {})},
        "endpoint_B": {"seed": sb, **(L.latent_stats(z_path[-1]) if z_path is not None else {})},
        "frames": frame_rows,
        "files": {
            "endpoint_A/B": "the two endpoint images (t = 0 and t = 1)",
            "intermediate/": "only the images produced between the endpoints",
            "all_frames/": "every frame in order, endpoints included",
            "interpolation_grid.png": "contact sheet of the whole path",
            "interpolation.gif": "animated walk A -> B -> A",
            "latents.npz": "t values, z_path (if Z-space) and w_path arrays",
        },
    }
    files["metadata.json"] = json.dumps(meta, indent=2).encode()
    return files


def save_run(files: dict[str, bytes], run_name: str) -> Path:
    out = OUTPUTS_DIR / run_name
    for rel, data in files.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return out


def zip_bytes(files: dict[str, bytes], root: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, data in files.items():
            z.writestr(f"{root}/{rel}", data)
    return buf.getvalue()


def run_name(params: dict) -> str:
    return f"run_{time.strftime('%Y%m%d_%H%M%S')}_seed{params['seed_a']}-to-{params['seed_b']}_{params['method']}"
