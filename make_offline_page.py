"""
Make ONE self-contained HTML file (real StyleGAN2 images + slider + charts) that opens
on any laptop by double-click - no Python, no internet.

    python make_offline_page.py                      # 4 default seed pairs
    python make_offline_page.py --pairs 42-1234 5-77 # your own pairs
"""
import argparse
from pathlib import Path

from lse import offline
from lse.generator import DEFAULT_MODEL, MODELS, LatentGenerator, download_weights

OUT = Path(__file__).resolve().parent / "Latent_Space_Explorer_OFFLINE.html"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", help="seed pairs like 42-1234")
    ap.add_argument("--psi", type=float, default=0.7)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--checkpoint")
    ap.add_argument("--untrained-demo", action="store_true", help="random untrained net (testing only)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    pairs = [tuple(int(x) for x in p.split("-")) for p in a.pairs] if a.pairs else None
    if a.untrained_demo:
        g = LatentGenerator.random_init(resolution=64)
    else:
        g = LatentGenerator.from_pickle(a.checkpoint or download_weights(a.model), name=a.model)
    html = offline.build_page(g, pairs, a.psi)
    Path(a.out).write_text(html, encoding="utf-8")
    print(f"\n[offline] saved {a.out}  ({len(html) / 1e6:.1f} MB)")
    print("[offline] copy this ONE file to any laptop and double-click it to open in the browser.")


if __name__ == "__main__":
    main()
