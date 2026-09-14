# Third-Party Notices

S2VC includes or adapts code and model weights from the projects listed below.
The S2VC license does not replace the licenses that apply to these components.
Source files retain their upstream notices where those notices were present.

## DMD2

- Local path: `src/modules/dmd`
- Upstream: https://github.com/tianweiy/DMD2
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
- License text: `LICENSE.md`

`src/modules/dmd/utils.py` contains adapted utility code from DMD2.

## Taming Transformers

- Local path: `src/modules/vqgan`
- Upstream: https://github.com/CompVis/taming-transformers
- License: MIT
- License text: `src/modules/vqgan/LICENSE`

`src/modules/vqgan/blocks.py` contains adapted VQGAN residual and attention
blocks.

## DCVC And Entropy Coding

- Local paths: `src/modules/dcvc.py`, `src/modules/entropy`, and `src/cpp`
- Upstream: https://github.com/microsoft/DCVC

The DCVC-derived files use multiple licenses:

- `src/modules/dcvc.py`, `src/cpp/py_rans/rans.cpp`, and
  `src/cpp/py_rans/rans.h`: Apache License 2.0. The files retain their
  InterDigital copyright and license notices. License text:
  https://www.apache.org/licenses/LICENSE-2.0
- `src/modules/entropy`, `src/cpp/setup.py`,
  `src/cpp/py_rans/py_rans.cpp`, and `src/cpp/py_rans/py_rans.h`: MIT. The
  files with upstream notices retain the Microsoft copyright and license
  notices. License text: https://opensource.org/license/mit
- `src/cpp/py_rans/rans_byte.h`: CC0 1.0 / public-domain dedication by
  Fabian Giesen, from https://github.com/rygorous/ryg_rans. License text:
  https://creativecommons.org/publicdomain/zero/1.0/legalcode

## FloLPIPS

- Local path: `src/modules/flolpips`
- Upstream: https://github.com/danier97/flolpips
- License: MIT
- License text: `src/modules/flolpips/LICENSE`

## Stable Diffusion Model Weights

The separately distributed `SD_ckpts.zip` contains Stable Diffusion model
weights. Use and redistribution of those weights are subject to their upstream
model licenses:

- Stable Diffusion 1.5: CreativeML Open RAIL-M, as documented by
  https://huggingface.co/runwayml/stable-diffusion-v1-5
- Stable Diffusion 2.1 VAE: CreativeML Open RAIL++-M, as documented by
  https://huggingface.co/stabilityai/stable-diffusion-2-1

Review the applicable model license before using or redistributing the
checkpoint archive.