"""
Command-line version (no browser): generate the two endpoint images and all
intermediate images, and save them to outputs/.

    python generate_cli.py --seed-a 42 --seed-b 1234 --steps 8 --method slerp
"""
import argparse

from lse import exporter
from lse.generator import DEFAULT_MODEL, MODELS, LatentGenerator, download_weights


def main():
    ap = argparse.ArgumentParser(description="Latent interpolation from the command line")
    ap.add_argument("--seed-a", type=int, default=42)
    ap.add_argument("--seed-b", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=8, help="frames including both endpoints")
    ap.add_argument("--method", default="slerp", choices=["slerp", "lerp", "w_lerp"])
    ap.add_argument("--psi", type=float, default=0.7, help="truncation psi")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--checkpoint", help="path to a StyleGAN2 .pkl (overrides --model)")
    a = ap.parse_args()

    path = a.checkpoint or download_weights(a.model)
    gen = LatentGenerator.from_pickle(path)
    params = dict(seed_a=a.seed_a, seed_b=a.seed_b, steps=a.steps, method=a.method, truncation_psi=a.psi)
    ts, imgs, zp, wp = gen.interpolate_images(a.seed_a, a.seed_b, a.steps, a.method, a.psi)
    files = exporter.build_files(gen.info(), params, ts, imgs, zp, wp)
    out = exporter.save_run(files, exporter.run_name(params))
    print(f"Saved {len(files)} files to {out}")
    print("  endpoint_A / endpoint_B  -> the two endpoint images")
    print("  intermediate/            -> images produced during interpolation")
    print("  interpolation_grid.png, interpolation.gif, latents.npz, metadata.json")


if __name__ == "__main__":
    main()
