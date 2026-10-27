"""
Backend: load a pretrained StyleGAN2 generator and turn latent vectors into images.

How a StyleGAN2 generator maps a latent vector to an image
----------------------------------------------------------
    seed --(RandomState)--> z  (512,)  ~ N(0, I)
    z    --(mapping network: 8-layer MLP)--> w (num_ws, 512)   "style" vectors
    w    --(truncation: w = w_avg + psi * (w - w_avg))--> w'   quality/diversity trade-off
    w'   --(synthesis network: modulated convs, 4x4 -> RxR)--> image (3, R, R) in [-1, 1]
    image --(x * 127.5 + 128, clip, uint8)--> RGB picture

The official NVIDIA network pickles (.pkl) are loaded with the vendored
``stylegan2/`` code (dnnlib, torch_utils, legacy.py from NVlabs/stylegan3).
On CPU the custom CUDA kernels are skipped automatically and the pure
PyTorch reference implementation is used, so no compiler is needed.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import latent as L

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Vendored NVIDIA code must be importable as top-level packages
# (network pickles reference modules such as ``torch_utils.persistence``).
_SG2_DIR = str(PROJECT_ROOT / "stylegan2")
if _SG2_DIR not in sys.path:
    sys.path.insert(0, _SG2_DIR)

_NGC = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/"


@dataclass(frozen=True)
class ModelInfo:
    key: str
    filename: str
    url: str
    description: str


# Small official StyleGAN2 checkpoints (256 px keeps CPU inference fast).
MODELS: "OrderedDict[str, ModelInfo]" = OrderedDict(
    (m.key, m)
    for m in [
        ModelInfo("ffhq-256", "stylegan2-ffhq-256x256.pkl", _NGC + "stylegan2-ffhq-256x256.pkl",
                  "Human faces (FFHQ), 256x256 - default, recommended"),
        ModelInfo("celebahq-256", "stylegan2-celebahq-256x256.pkl", _NGC + "stylegan2-celebahq-256x256.pkl",
                  "Celebrity faces (CelebA-HQ), 256x256"),
        ModelInfo("lsundog-256", "stylegan2-lsundog-256x256.pkl", _NGC + "stylegan2-lsundog-256x256.pkl",
                  "Dogs (LSUN Dog), 256x256"),
        ModelInfo("afhqcat-512", "stylegan2-afhqcat-512x512.pkl", _NGC + "stylegan2-afhqcat-512x512.pkl",
                  "Cats (AFHQ Cat), 512x512 - slower on CPU"),
    ]
)
DEFAULT_MODEL = "ffhq-256"


# --------------------------------------------------------------------------
# Weight download / cache
# --------------------------------------------------------------------------
def weights_path(key: str) -> Path:
    return WEIGHTS_DIR / MODELS[key].filename


def download_weights(key: str = DEFAULT_MODEL, force: bool = False) -> Path:
    """Download a pretrained checkpoint into ``weights/`` once and reuse it (cached weights)."""
    import requests  # imported lazily: only needed for the first download

    info = MODELS[key]
    dest = weights_path(key)
    if dest.exists() and dest.stat().st_size > 1_000_000 and not force:
        print(f"[weights] cached: {dest.name}")
        return dest

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"[weights] downloading {info.filename}\n          from {info.url}")
    with requests.get(info.url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r          {done / 1e6:7.1f} / {total / 1e6:.1f} MB  ({pct:5.1f}%)", end="", flush=True)
    print()
    tmp.replace(dest)
    print(f"[weights] saved to {dest}")
    return dest


def find_local_checkpoints() -> list[Path]:
    return sorted(WEIGHTS_DIR.glob("*.pkl"))


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
def tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    """(N, C, H, W) float in ~[-1, 1]  ->  (N, H, W, C) uint8 in [0, 255]."""
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    arr = img.cpu().numpy()
    if arr.shape[-1] == 1:  # grayscale models
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def to_data_url(arr: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(to_png_bytes(arr)).decode("ascii")


# --------------------------------------------------------------------------
# Generator wrapper
# --------------------------------------------------------------------------
class LatentGenerator:
    """Thin, testable wrapper around a StyleGAN2 ``G`` network."""

    def __init__(self, G: torch.nn.Module, name: str = "custom", device: str = "cpu", pretrained: bool = True):
        self.G = G.eval().requires_grad_(False).to(device)
        self.name = name
        self.device = torch.device(device)
        self.pretrained = pretrained
        self.z_dim = int(G.z_dim)
        self.w_dim = int(G.w_dim)
        self.c_dim = int(getattr(G, "c_dim", 0))
        self.num_ws = int(G.num_ws)
        self.resolution = int(G.img_resolution)
        self._lock = threading.Lock()  # torch modules are not re-entrant across Flask threads

    # ---------------- loading ----------------
    @classmethod
    def from_pickle(cls, path: str | os.PathLike, device: str = "cpu", name: str | None = None) -> "LatentGenerator":
        import dnnlib  # noqa: F401  (vendored)
        import legacy  # vendored NVIDIA loader

        path = Path(path)
        print(f"[model] loading {path.name} on {device} ...")
        with dnnlib.util.open_url(str(path)) as f:
            G = legacy.load_network_pkl(f)["G_ema"]
        gen = cls(G, name=name or path.stem, device=device, pretrained=True)
        print(f"[model] ready: {gen.resolution}x{gen.resolution}, z_dim={gen.z_dim}, w_dim={gen.w_dim}, num_ws={gen.num_ws}")
        return gen

    @classmethod
    def random_init(cls, resolution: int = 32, z_dim: int = 512, w_dim: int = 512, seed: int = 0) -> "LatentGenerator":
        """An *untrained* StyleGAN2 with random weights. Used only by the unit tests
        (and the optional ``--untrained-demo`` flag) so the pipeline can be checked offline."""
        from training.networks_stylegan2 import Generator

        torch.manual_seed(seed)
        G = Generator(z_dim=z_dim, c_dim=0, w_dim=w_dim, img_resolution=resolution, img_channels=3,
                      mapping_kwargs={"num_layers": 2}, channel_base=2048, channel_max=64)
        return cls(G, name=f"untrained-random-{resolution}px", pretrained=False)

    # ---------------- core mapping ----------------
    def _label(self, n: int) -> torch.Tensor | None:
        if self.c_dim == 0:
            return None
        c = torch.zeros(n, self.c_dim, device=self.device)
        c[:, 0] = 1  # conditional models: use class 0
        return c

    @torch.no_grad()
    def z_to_w(self, z: np.ndarray, truncation_psi: float = 0.7) -> np.ndarray:
        """Mapping network. z: (N, z_dim) or (z_dim,)  ->  w: (N, num_ws, w_dim)."""
        z = np.atleast_2d(np.asarray(z, dtype=np.float32))
        if z.shape[1] != self.z_dim:
            raise ValueError(f"expected z with {self.z_dim} dims, got {z.shape[1]}")
        zt = torch.from_numpy(z).to(self.device)
        with self._lock:
            w = self.G.mapping(zt, self._label(len(z)), truncation_psi=float(truncation_psi))
        return w.cpu().numpy()

    @torch.no_grad()
    def w_to_image(self, w: np.ndarray) -> np.ndarray:
        """Synthesis network. w: (N, num_ws, w_dim) or (w_dim,) -> (N, H, W, 3) uint8."""
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 1:  # a single 512-d w: broadcast to every layer
            w = np.tile(w[None, None, :], (1, self.num_ws, 1))
        elif w.ndim == 2:
            w = w[None]
        wt = torch.from_numpy(w).to(self.device)
        with self._lock:
            img = self.G.synthesis(wt, noise_mode="const")  # const noise => deterministic output
        return tensor_to_uint8(img)

    def z_to_image(self, z: np.ndarray, truncation_psi: float = 0.7) -> np.ndarray:
        return self.w_to_image(self.z_to_w(z, truncation_psi))

    def seed_to_image(self, seed: int, truncation_psi: float = 0.7) -> np.ndarray:
        return self.z_to_image(L.seed_to_z(seed, self.z_dim), truncation_psi)[0]

    # ---------------- interpolation ----------------
    def latent_at(self, seed_a: int, seed_b: int, t: float, method: str, psi: float):
        """Return (z_t or None, w_t) for a point on the path. For Z-space methods the
        z vector is interpolated and then mapped; for ``w_lerp`` the w vectors are."""
        za, zb = L.seed_to_z(seed_a, self.z_dim), L.seed_to_z(seed_b, self.z_dim)
        if method == "w_lerp":
            wa, wb = self.z_to_w(np.stack([za, zb]), psi)
            return None, L.lerp(wa, wb, t)
        zt = L.interpolate(za, zb, t, method)
        return zt, self.z_to_w(zt, psi)[0]

    def interpolate_images(self, seed_a: int, seed_b: int, steps: int = 8,
                           method: str = "slerp", truncation_psi: float = 0.7, batch: int = 4):
        """Generate the full path: returns (ts, images[N,H,W,3], z_path or None, w_path[N,num_ws,w_dim])."""
        if method not in L.INTERPOLATION_METHODS:
            raise ValueError(f"method must be one of {L.INTERPOLATION_METHODS}")
        ts = L.interpolation_ts(steps)
        za, zb = L.seed_to_z(seed_a, self.z_dim), L.seed_to_z(seed_b, self.z_dim)
        if method == "w_lerp":
            wa, wb = self.z_to_w(np.stack([za, zb]), truncation_psi)
            z_path = None
            w_path = np.stack([L.lerp(wa, wb, float(t)) for t in ts])
        else:
            z_path = L.interpolation_path(za, zb, steps, method)
            w_path = self.z_to_w(z_path, truncation_psi)
        imgs = [self.w_to_image(w_path[i:i + batch]) for i in range(0, len(w_path), batch)]
        return ts, np.concatenate(imgs), z_path, w_path

    def info(self) -> dict:
        return {
            "name": self.name, "pretrained": self.pretrained, "resolution": self.resolution,
            "z_dim": self.z_dim, "w_dim": self.w_dim, "num_ws": self.num_ws, "c_dim": self.c_dim,
            "device": str(self.device),
            "params_mapping": sum(p.numel() for p in self.G.mapping.parameters()),
            "params_synthesis": sum(p.numel() for p in self.G.synthesis.parameters()),
        }


class FrameCache:
    """Tiny LRU cache so dragging the slider back over a frame is instant."""

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)
