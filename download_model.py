"""
Download (and cache) a pretrained StyleGAN2 checkpoint into weights/.

    python download_model.py               # default: FFHQ faces 256x256
    python download_model.py --model lsundog-256
    python download_model.py --list
"""
import argparse
import sys

from lse.generator import DEFAULT_MODEL, MODELS, LatentGenerator, download_weights


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--list", action="store_true", help="show available models")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--no-verify", action="store_true", help="skip the test load after downloading")
    args = ap.parse_args()

    if args.list:
        for k, m in MODELS.items():
            print(f"  {k:14s} {m.description}")
        return

    try:
        path = download_weights(args.model, force=args.force)
    except Exception as e:
        info = MODELS[args.model]
        print(f"\n[weights] DOWNLOAD FAILED: {e}\n")
        print("Manual fix: open this link in a browser, save the file, and put it in the 'weights' folder:")
        print(f"  {info.url}\n  (file name: {info.filename})")
        sys.exit(1)

    if not args.no_verify:
        gen = LatentGenerator.from_pickle(path)
        img = gen.seed_to_image(42)
        print(f"[verify] OK - generated a {img.shape[1]}x{img.shape[0]} test image from seed 42")


if __name__ == "__main__":
    main()
