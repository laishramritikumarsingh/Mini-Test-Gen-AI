# Vendored StyleGAN2 loader code

Copied from https://github.com/NVlabs/stylegan3 (which also loads StyleGAN2 network
pickles), under the NVIDIA Source Code License in `LICENSE-NVIDIA.txt`
(non-commercial / research and academic use).

Only the files needed to load and run a pretrained generator are included:
`dnnlib/`, `torch_utils/`, `legacy.py`, `training/networks_stylegan2.py`.

Small compatibility patches (marked `[patched]` in the source) so it runs on
modern Python / PyTorch:

* `torch_utils/ops/conv2d_gradfix.py`, `torch_utils/ops/grid_sample_gradfix.py` -
  removed the `pkg_resources` import (not available in new Python environments).
* `dnnlib/util.py` - replaced `distutils.util.strtobool` (distutils was removed in
  Python 3.12) with a local function.

On CPU the custom CUDA kernels are never compiled; the built-in pure-PyTorch
reference implementation is used automatically.
