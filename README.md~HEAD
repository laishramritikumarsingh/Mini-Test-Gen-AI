# Latent Space Explorer

An interactive application that explores **how changing a latent vector changes a generated image**, using a **pretrained StyleGAN2** generator.

Pick two seeds → each becomes a 512‑dimensional latent vector **z** → the app walks from one vector to the other and shows every image the generator produces along the way, together with live visualisations of what is happening inside the latent space.

---

## Quick start (on any laptop)

> Needs **Python 3.9 or newer** and an internet connection **the first time only** (to install packages and download the model weights). No GPU needed — it runs on CPU.

### Windows
1. Unzip the folder (right‑click → *Extract All*). Don't run it from inside the zip.
2. Double‑click **`START_WINDOWS.bat`**.
3. Wait for the first‑time setup (a few minutes). The browser opens at **http://127.0.0.1:8765**.

If Python isn't installed: get it from <https://www.python.org/downloads/> and **tick "Add python.exe to PATH"** in the installer.

### macOS
Double-click **`START_MAC.command`** (if macOS blocks it: right-click → Open → Open). Or in Terminal:

### macOS / Linux (Terminal)
```bash
cd latent-space-explorer
bash start_mac_linux.sh
```
The browser opens at **http://127.0.0.1:8765**.

Next time, the same script starts instantly (packages and weights are cached).

---

## Opening it on another laptop with **no Python** (double‑click)

The first time the start script runs, it also creates **`Latent_Space_Explorer_OFFLINE.html`** in this folder (takes 1–3 minutes once).

* That **one file** holds real pretrained‑StyleGAN2 images for 4 seed pairs × 3 interpolation methods (25 frames each), the slider, the filmstrip and all the charts.
* Copy it to any laptop (pen drive, WhatsApp, email, Google Drive) and **double‑click it**. It opens in the browser, no Python and no internet needed.
* From inside the running app, **Save offline page (.html)** makes the same kind of file for the seeds you are looking at.
* Or build it yourself: `python make_offline_page.py --pairs 42-1234 7-2026`

The offline page only contains the pre‑generated seed pairs. To try any seed, change ψ or the number of frames, run the full app.

---

## How the assignment requirements are met

| Requirement | Where |
|---|---|
| **Frontend: seed input** | Seed A / Seed B fields (with random 🎲 and swap ⇅ buttons) |
| **Frontend: latent interpolation slider** | The `t` slider under the images (0 = A, 1 = B). Dragging shows the nearest frame instantly, then renders the **exact** latent at that `t` (badge "exact t"). ▶ plays the walk. |
| **Frontend: image display** | Endpoint A, current image, endpoint B, plus the filmstrip of every frame |
| **Backend: load a pretrained StyleGAN2** | `lse/generator.py` loads NVIDIA's official StyleGAN2 network pickle (`stylegan2-ffhq-256x256.pkl` by default) |
| **Backend: generate images from latent vectors** | `z_to_w()` (mapping network) → `w_to_image()` (synthesis network), served by Flask in `app.py` |
| **Output: two endpoint images + intermediates** | Shown in the UI; **Export outputs** saves `endpoint_A_*.png`, `endpoint_B_*.png`, `intermediate/frame_*.png`, a labelled grid, a GIF, the latent vectors and `metadata.json` |
| **Test cases: latent representations** | `tests/test_latent_representations.py` |
| **Test cases: interpolation** | `tests/test_interpolation.py` |
| **Test cases: generator mapping z → image** | `tests/test_generator_mapping.py` |
| **Cached weights / small pretrained model** | 256×256 StyleGAN2 checkpoint, downloaded once and cached in `weights/` |

---

## Using the app

1. **Seeds** – type two numbers (0 – 4,294,967,295) or press 🎲.
2. **Method**
   - **Slerp (Z)** – spherical interpolation in Z space (default). Keeps the vector on the hypersphere where Gaussian latents live, so mid‑frames look natural.
   - **Lerp (Z)** – straight line in Z. The midpoint vector is shorter than normal (see the norm chart) and can look washed‑out / averaged.
   - **Lerp (W)** – maps both seeds through the mapping network first, then blends the style vectors **w**. Usually the smoothest transition.
3. **Frames** – how many images along the path (including both endpoints).
4. **Truncation ψ** – lower = closer to the "average" image, cleaner but less varied.
5. **Generate interpolation** → drag the slider, click any filmstrip frame, or press ▶.
6. **Export outputs (.zip)** – downloads every output and also saves it in `outputs/`.

### Visualisations
* **Latent heat strip** – the first 48 of 512 dimensions of z_A, z(t), z_B. The middle row morphs live with the slider.
* **Latent norm ‖z(t)‖** – why slerp is preferred: lerp dips well below the expected radius √512 ≈ 22.6.
* **Similarity to each endpoint** – cosine similarity of every frame to A and to B (crossing near t = 0.5).
* **Image change per step** – pixel change between consecutive frames; even bars = smooth walk. Click a bar to jump there.
* **Distribution of z values** – histogram of the 512 numbers in z_A against the standard normal curve.
* **Endpoint statistics** and the **angle** between z_A and z_B (≈ 90° — random high‑dimensional vectors are nearly orthogonal).
* **Pipeline panel** – Seed → z → mapping → w → truncation → synthesis → image, with the real shapes of the loaded model.

All charts have hover tooltips and work in light and dark mode (◐ button).

---

## How the generator maps a latent vector to an image

```
seed ──RandomState──▶ z ∈ R^512 ~ N(0, I)
z    ──mapping network (8‑layer MLP)──▶ w ∈ R^512  (copied to every synthesis layer → shape [14, 512] at 256 px)
w    ──truncation: w' = w̄ + ψ (w − w̄)──▶ w'
w'   ──synthesis network (style‑modulated convs, 4×4 → 256×256)──▶ image in [−1, 1]
image ──×127.5 + 128, clip──▶ RGB uint8
```

Noise inputs are fixed (`noise_mode="const"`) so the same latent always gives exactly the same image.

---

## Project structure

```
latent-space-explorer/
├── START_WINDOWS.bat          one‑click setup + run (Windows)
├── start_mac_linux.sh         one‑click setup + run (macOS / Linux)
├── RUN_TESTS_WINDOWS.bat      run the test suite (Windows)
├── run_tests.sh               run the test suite (macOS / Linux)
├── app.py                     Flask backend + REST API
├── generate_cli.py            command‑line version (no browser)
├── download_model.py          downloads / caches pretrained weights
├── make_offline_page.py       builds the double-click offline HTML file
├── START_MAC.command          double-click launcher (macOS)
├── requirements.txt
├── lse/
│   ├── latent.py              seed → z, lerp, slerp, interpolation paths, stats
│   ├── generator.py           loads StyleGAN2, z → w → image, caching
│   ├── exporter.py            saves endpoints, intermediates, grid, GIF, metadata
│   ├── payload.py             JSON data for the frontend
│   └── offline.py             builds the self-contained offline HTML page
├── static/                    frontend (index.html, style.css, app.js, charts.js – no internet needed)
├── stylegan2/                 NVIDIA's official loader code (dnnlib, torch_utils, legacy.py) – see PATCHES.md
├── tests/                     pytest test cases
├── weights/                   cached .pkl model (created on first run)
└── outputs/                   exported runs
```

### API
| Endpoint | Purpose |
|---|---|
| `GET /api/status` | model state, resolution, z/w dimensions, available models |
| `POST /api/interpolate` | `{seed_a, seed_b, steps, method, truncation_psi}` → all frames + latent data for the charts |
| `POST /api/frame` | same + `t` → image at any exact slider position |
| `POST /api/export` | zip of all outputs (also saved in `outputs/`) |
| `POST /api/offline_page` | single self-contained HTML file for the current seeds |
| `POST /api/load_model` | switch pretrained model |

---

## Running the tests

Windows: double‑click `RUN_TESTS_WINDOWS.bat` · macOS/Linux: `bash run_tests.sh`

or manually: `.venv/bin/python -m pytest` (Windows: `.venv\Scripts\python -m pytest`)

* **45 tests**. 43 run offline in about a second using a tiny randomly‑initialised StyleGAN2 (same architecture, untrained) so the mapping logic is checked without the download.
* The last 2 run against the **real pretrained model** when it is in `weights/` (they are skipped otherwise).

What is tested:
* **Latent representations** – shape 512, determinism per seed, matches NVIDIA's seed convention, mean ≈ 0 / std ≈ 1, norm ≈ √512, random latents ≈ orthogonal, invalid seeds rejected.
* **Interpolation** – exact endpoints, lerp midpoint = average, lerp norm dip, slerp preserves norm, slerp angle moves uniformly, path moves monotonically toward B, edge cases (parallel/identical vectors, bad t, bad method).
* **Generator mapping** – z → w shape `[num_ws, 512]`, same w broadcast to every layer, truncation ψ = 0 collapses to the average, ψ scales distance linearly, image shape/dtype/range, determinism, different latents → different images, batch = single, small latent change → small image change, interpolation endpoints equal the seed images, exported files contain endpoints + intermediates.

---

## Command line (optional)

```bash
.venv/bin/python generate_cli.py --seed-a 42 --seed-b 1234 --steps 10 --method slerp
```
Windows: `.venv\Scripts\python generate_cli.py --seed-a 42 --seed-b 1234 --steps 10`

Other app options: `python app.py --help` (`--model lsundog-256`, `--checkpoint path.pkl`, `--port 9000`, `--no-browser`).

Available pretrained models (switchable from the dropdown at the top): FFHQ faces 256 (default), CelebA‑HQ faces 256, LSUN dogs 256, AFHQ cats 512 (slower on CPU).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Python was not found" (Windows) | Install Python from python.org, tick **Add python.exe to PATH**, re‑run. |
| Weights download fails (no internet / blocked network) | Open this link in a browser and save the file into the `weights` folder, then restart: `https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl` |
| Port already in use | `python app.py --port 9000` (or run the start script with `--port 9000`) |
| Browser didn't open | Open **http://127.0.0.1:8765** manually. |
| Want to check the app without downloading | `python app.py --untrained-demo` – runs a random, **untrained** StyleGAN2 (images are just colour noise; for testing only). |
| Packages broken / want a clean reinstall | Delete the `.venv` folder and run the start script again. |
| Slow generation | Use fewer frames, or keep the 256×256 model. On a typical laptop CPU it takes roughly 0.3 – 1 s per 256×256 image. |

To show the project **without internet** on the demo day, run the start script once beforehand on that laptop (with internet) – after that everything is cached in `.venv/` and `weights/`.

---

## Credits

* StyleGAN2 – Karras et al., *Analyzing and Improving the Image Quality of StyleGAN* (CVPR 2020).
* Pretrained weights and loader code – NVIDIA (NVlabs/stylegan3), NVIDIA Source Code License (research / non‑commercial use). See `stylegan2/LICENSE-NVIDIA.txt` and `stylegan2/PATCHES.md`.
