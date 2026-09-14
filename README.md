## S²VC: Single-step Diffusion-based Video Coding with Semantic-Temporal Guidance

[![arXiv](https://img.shields.io/badge/arXiv-2512.07480-b31b1b.svg)](https://arxiv.org/abs/2512.07480)
[![project](https://img.shields.io/badge/Project-Page-orange)](https://onedc-codec.github.io/s2vc/)
[![python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

Naifu Xue, Zhaoyang Jia, Jiahao Li, Bin Li, Zihan Zheng, Yuan Zhang, Yan Lu

⭐ If you find S²VC helpful, please consider starring this repository. Thank you! 🤗


## 👍 More Works
- [One-Step Diffusion-Based Image Compression with Semantic Distillation (NeurIPS 2025)](https://github.com/onedc-codec/onedc)
- [DLF: Extreme Image Compression with Dual-generative Latent Fusion (ICCV 2025 Highlight)](https://github.com/Xiaosu-Zhu/DLF)
- [Generative Latent Coding for Ultra-Low Bitrate Image Compression (CVPR 2024)](https://github.com/jzyustc/GLC)


## 📝 Abstract
While traditional and neural video codecs (NVCs) have achieved remarkable rate–distortion performance, improving perceptual quality at low bitrates remains challenging. Some NVCs incorporate perceptual or adversarial objectives but still suffer from artifacts due to limited generation capacity, whereas others leverage pretrained diffusion models to improve quality at the cost of high sampling complexity. To overcome these challenges, we propose **S²VC**, a **S**ingle-**S**tep diffusion–based **V**ideo **C**odec that integrates a conditional coding framework with an efficient single-step diffusion generator, enabling realistic reconstruction at low bitrates with reduced sampling cost. Recognizing the importance of semantic conditioning in single-step diffusion, we introduce *Contextual Semantic Guidance* to extract frame-adaptive semantics from buffered features. This guidance replaces text captions with efficient, fine-grained conditioning, thereby improving generation realism. In addition, *Temporal Consistency Guidance* is incorporated into the diffusion U-Net to enforce temporal coherence across frames and ensure stable generation. Extensive experiments show that S²VC delivers state-of-the-art perceptual quality with an average bitrate saving of 51.62% over prior perceptual method, underscoring the promise of single-step diffusion for efficient, high-quality video compression.


## 💿 Installation

**Prerequisites**
- Linux (tested on Ubuntu 22.04)
- Python 3.10
- CUDA-capable GPU (≥ 24 GB VRAM recommended for 1080p)
- A C++17 compiler (gcc ≥ 7 / clang ≥ 5)

The released `SD_ckpts.zip` contains the required Stable Diffusion 1.5 model
and Stable Diffusion 2.1 VAE, so inference does not require HuggingFace access
after the release archives have been downloaded and extracted.

**1. Create environment & install Python dependencies**
```bash
conda create -n s2vc python=3.10 -y
conda activate s2vc

# Install torch first, matched to your CUDA driver. requirements.txt
# deliberately omits torch so it doesn't pull in a wheel that mismatches your
# driver. Example for a CUDA 12.x driver (check `nvidia-smi`):
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

The diffusers / peft / transformers / accelerate versions in `requirements.txt`
are pinned to a known-good set that matches the LoRA hooks used at training
time; upgrading these on top of the released checkpoints may break loading.

**2. Build the C++ rANS entropy coder**
```bash
cd src/cpp
pip install -e .
cd ../..
```


## 💾 Pre-trained Checkpoints

The checkpoints are distributed as separate ZIP archives in the
[OneDrive release folder](https://1drv.ms/f/c/af332a47fcf136b4/IgB7n-zqVtLKR6974H0FSWg-AWXdZ9FUlrjfdPOFcrsYQIw?e=MnYkya).
This lets you download only the rate points you need.

Always download `SD_ckpts.zip`. For each video λ you want to run, also download
the corresponding `P_ckpts_lmbda<L>.zip` and the paired
`I_ckpts_lmbda<L>.zip` listed in the Model Pairs table below. The complete
release contains:

```text
SD_ckpts.zip
I_ckpts_lmbda1.8.zip
I_ckpts_lmbda2.9.zip
I_ckpts_lmbda4.6.zip
I_ckpts_lmbda7.4.zip
I_ckpts_lmbda12.2.zip
P_ckpts_lmbda0.2.zip
P_ckpts_lmbda0.4.zip
P_ckpts_lmbda0.6.zip
P_ckpts_lmbda1.2.zip
P_ckpts_lmbda1.8.zip
P_ckpts_lmbda2.6.zip
P_ckpts_lmbda3.7.zip
P_ckpts_lmbda5.2.zip
```

Place the downloaded ZIP files in `runs/` and extract them there:

```bash
mkdir -p runs
cd runs
for archive in *.zip; do unzip -n "$archive"; done
cd ..
```

Each archive already contains its `I_ckpts/`, `P_ckpts/`, or `SD_ckpts/`
top-level directory. After extracting all archives, the layout is:

```
runs/
├── I_ckpts/
│   ├── onedc_lmbda1.8/
│   │   └── model_1.safetensors    # 1.58 GB — IntraNoAR state_dict
│   ├── onedc_lmbda2.9/
│   ├── onedc_lmbda4.6/
│   ├── onedc_lmbda7.4/
│   └── onedc_lmbda12.2/
├── P_ckpts/
│   ├── lmbda0.2/
│   │   ├── model.safetensors      # 4.88 GB — SD1.5 UNet + 2× LoRA + codec hooks
│   │   └── model_1.safetensors    # 576 MB — DMC_OneDC P-frame codec state_dict
│   ├── lmbda0.4/
│   ├── lmbda0.6/
│   ├── lmbda1.2/
│   ├── lmbda1.8/
│   ├── lmbda2.6/
│   ├── lmbda3.7/
│   └── lmbda5.2/
└── SD_ckpts/
    ├── stable-diffusion-v1-5/
    │   ├── unet/
    │   ├── scheduler/
    │   ├── vae/
    │   ├── text_encoder/
    │   ├── tokenizer/
    │   └── ...
    └── stable-diffusion-2-1/
        └── vae/
            ├── config.json
            └── diffusion_pytorch_model.safetensors
```

Point `configs/inference.yaml` at the extracted SD directories. The paths below
assume inference is launched from `src/` as shown in the Inference section;
absolute paths also work:

```yaml
sd15: ../runs/SD_ckpts/stable-diffusion-v1-5
sd21_vae: ../runs/SD_ckpts/stable-diffusion-2-1
```

`stable-diffusion-v1-5/` is a complete Diffusers model directory. Only the
Stable Diffusion 2.1 VAE is required by S²VC, so
`stable-diffusion-2-1/` intentionally contains only `vae/`.

The release ships **8 rate points** spanning the low-bitrate operating regime
targeted by the paper. See the next section for the I-frame ↔ P-frame pairing.


## 📋 Model Pairs

The training loss is `D + λ·R`, so **higher video λ → lower bpp** (smaller files,
lower quality); **lower video λ → higher bpp** (bigger files, better quality).

The I-frame and P-frame codecs were trained independently, then paired at the
end of training; the pairing must be honoured at inference time. Using a
mismatched I-frame ckpt will degrade RD performance.

| Video λ | I-frame ckpt                                       | P-frame ckpt folder       |
|---------|----------------------------------------------------|---------------------------|
| 0.2     | `runs/I_ckpts/onedc_lmbda1.8/model_1.safetensors`  | `runs/P_ckpts/lmbda0.2/`  |
| 0.4     | `runs/I_ckpts/onedc_lmbda1.8/model_1.safetensors`  | `runs/P_ckpts/lmbda0.4/`  |
| 0.6     | `runs/I_ckpts/onedc_lmbda1.8/model_1.safetensors`  | `runs/P_ckpts/lmbda0.6/`  |
| 1.2     | `runs/I_ckpts/onedc_lmbda2.9/model_1.safetensors`  | `runs/P_ckpts/lmbda1.2/`  |
| 1.8     | `runs/I_ckpts/onedc_lmbda2.9/model_1.safetensors`  | `runs/P_ckpts/lmbda1.8/`  |
| 2.6     | `runs/I_ckpts/onedc_lmbda4.6/model_1.safetensors`  | `runs/P_ckpts/lmbda2.6/`  |
| 3.7     | `runs/I_ckpts/onedc_lmbda7.4/model_1.safetensors`  | `runs/P_ckpts/lmbda3.7/`  |
| 5.2     | `runs/I_ckpts/onedc_lmbda12.2/model_1.safetensors` | `runs/P_ckpts/lmbda5.2/`  |

The I-frame λ does **not** equal the video λ — they are independent operating
points on their respective rate-distortion curves (image vs. video).

See [`model_pair.md`](./runs/model_pair.md) for the same table in standalone form.


## 📁 Input Data Format

The inference script reads videos as **directories of PNG frames** (one
directory per sequence). It does *not* accept YUV, MP4, or other container
formats directly — pre-extract the frames if you have those.

**Directory layout** — `--video_ds_folder` points at a root containing one
subdirectory per sequence; the subdirectory name becomes the sequence
identifier in the output:

```
<video_ds_folder>/
├── <sequence_A>/
│   ├── im00001.png
│   ├── im00002.png
│   └── ...
├── <sequence_B>/
│   ├── 0001.png
│   └── ...
└── ...
```

Alternatively, `--video_ds_folder` can be a plain `.txt` file listing one
sequence directory per line.

**Frame requirements**
- **Format**: 8-bit RGB PNG. Anything PIL can `convert('RGB')` works; alpha is dropped.
- **Naming**: any filename; frames are sorted by natural key, so `im00001.png`, `im00002.png`, ... and `0001.png`, `0002.png`, ... both work.
- **Resolution**: arbitrary. The inference path reflect-pads each frame to the nearest multiple of 64 internally; bpp is computed against the **original** H × W so padding does not bias the rate measurement.
- **Per-sequence consistency**: all frames within one sequence must share the same H × W.
- **Frame count**: the inference script reads the first `--n_frame` frames per sequence (default 96).


## 💻 Inference

`src/inference.py` encodes/decodes **one rate point** over every sequence under
`--video_ds_folder` and writes the reconstructed frames + an MP4 per sequence.

```bash
cd src
python inference.py \
    --config_path      ../configs/inference.yaml \
    --output_base_path ../output \
    --i_ckpt_path      ../runs/I_ckpts/onedc_lmbda2.9/model_1.safetensors \
    --ckpt_path        ../runs/P_ckpts/lmbda1.8 \
    --ds_name          UVG \
    --qp_name          lmbda1.8 \
    --video_ds_folder  /path/to/UVG_pngs \
    --n_frame 96 \
    --n_reset 32
```

The `--i_ckpt_path` and `--ckpt_path` arguments must follow the pairings in
the Model Pairs table above.

**CLI arguments**

| Argument | Required | Description |
|---|---|---|
| `--config_path` | yes | Architecture config; always `configs/inference.yaml` for the released checkpoints. |
| `--output_base_path` | yes | Root directory for reconstructed PNGs and MP4s. |
| `--i_ckpt_path` | yes | Absolute path to `model_1.safetensors` inside an `I_ckpts/onedc_lmbda<L>/` folder. |
| `--ckpt_path` | yes | Folder containing `model.safetensors` + `model_1.safetensors` for the P-frame codec (i.e. `P_ckpts/lmbda<L>/`). |
| `--ds_name` | yes | Dataset name; used as a top-level subdirectory under `--output_base_path`. |
| `--qp_name` | yes | Rate-point tag; used as a subdirectory inside each sequence's output folder. Recommend `lmbda<L>`. |
| `--video_ds_folder` | yes | Root directory of PNG sequences or a `.txt` file listing them. |
| `--n_frame` | no (default 96) | First N frames of each sequence to encode. |
| `--n_reset` | no (default 32) | DPB reset interval: every N P-frames, the reference is re-anchored to the I-frame's latent. |

**Output structure**

```
<output_base_path>/<ds_name>/
├── <sequence_A>/
│   └── <qp_name>/
│       ├── im00001_bpp_<i_bpp>.png        # I-frame reconstruction
│       ├── im00002_bpp_<p1_bpp>.png       # P-frame 1
│       ├── ...
│       └── im00096_bpp_<...>.png
├── <sequence_B>/
│   └── <qp_name>/
│       └── ...
└── recon_mp4/
    └── <qp_name>/
        ├── <sequence_A>.mp4
        └── <sequence_B>.mp4
```

Each frame's per-frame bpp (measured against the **original** H × W, not the
internally-padded H × W) is encoded into its filename.


## 📊 Quality Evaluation

`src/eval_quality.py` consumes inference's output directly and reports the
metrics used in the paper: **PSNR**, **MS-SSIM**, **LPIPS**, **DISTS**,
**FloLPIPS**, and patch-based **FID** (FID/256 per Mentzer et al. 2020).
Per-frame **bpp** is also reported, read from the recon PNG filenames.

The four "shared" arguments map 1:1 to the corresponding `inference.py` args —
just pass the same values you used for inference:

```bash
cd src
python eval_quality.py \
    --label_root      /path/to/UVG_pngs \
    --recon_base_path ../output \
    --ds_name         UVG \
    --qp_name         lmbda1.8 \
    --output_path     ../output/quality_csv
```

**CLI arguments**

| Argument | Required | Description |
|---|---|---|
| `--label_root` | yes | Root directory of per-sequence GT subdirectories. Same value as inference's `--video_ds_folder`. |
| `--recon_base_path` | yes | Same value as inference's `--output_base_path`. |
| `--ds_name` | yes | Same value as inference's `--ds_name`. |
| `--qp_name` | yes | Same value as inference's `--qp_name`. |
| `--output_path` | yes | Directory to write the resulting CSV files. |
| `--save_pfx` | no (default `""`) | Optional prefix prepended to output CSV filenames. |
| `--device` | no (default `cuda:0`) | Torch device. |
| `--fid_patch_size` | no (default 256) | Patch size for FID/256. Must be ≤ min(H, W); pass a smaller value (e.g. `--fid_patch_size 128`) when evaluating videos shorter than 256 px on the short edge (e.g. HEVC Class D/E). |
| `--fid_patch_num` | no (default 2) | Number of split-shifted patch grids (1 = single grid; 2 = also a `patch/2` shifted grid). |

**Input requirements** — the GT frame naming is **stricter** than inference's:
each label sequence must contain files matching `im{N:05d}.png` (e.g.
`im00001.png`). Permissive names that inference accepts (`0001.png`,
`frame_001.png`, ...) are **rejected** by `eval_quality.py`. Recon filenames
follow the `im{N:05d}_bpp_{bpp:.6f}.png` pattern that inference writes — no
preparation needed.

**Output** — for each `<ds_name>`, three CSVs are written under `--output_path/`:

```
<output_path>/
├── detail/
│   └── [<save_pfx>_]<seq>_detail.csv     # per-frame metrics for each sequence
├── [<save_pfx>_]<ds_name>_detail.csv     # per-sequence mean of frame-level metrics
└── [<save_pfx>_]<ds_name>_summary.csv    # dataset-level mean + patch-FID
```

| CSV | Columns |
|---|---|
| `<seq>_detail.csv` | `lpips`, `dists`, `psnr`, `ms_ssim`, `flolpips`, `bpp` (one row per frame) |
| `<ds_name>_detail.csv` | `name`, `psnr`, `msssim`, `lpips`, `dists`, `flolpips`, `bpp` (one row per sequence) |
| `<ds_name>_summary.csv` | dataset means of the per-sequence columns, plus `fid` |

The first invocation downloads metric backbones from PyTorch Hub / package
mirrors (roughly 900 MB total, cached for subsequent runs), including AlexNet,
VGG16, RAFT-large, and the Inception weights used for FID.


## 🥰 Acknowledgement
We sincerely thank the following outstanding works, which greatly inspired and supported our research:
- [DCVC family](https://github.com/microsoft/DCVC) — conditional coding framework and rANS entropy coder
- [DMD2: Improved Distribution Matching Distillation for Fast Image Synthesis](https://github.com/tianweiy/DMD2) — single-step diffusion generator
- [DINOv3](https://github.com/facebookresearch/dinov3) — semantic teacher for Contextual Semantic Guidance
- [FloLPIPS](https://github.com/danier97/flolpips) — temporal-aware perceptual metric


## 📕 Citation

If you find our work inspiring, please cite:
```bibtex
@InProceedings{Xue_2026_CVPR,
    author    = {Xue, Naifu and Jia, Zhaoyang and Li, Jiahao and Li, Bin and Zheng, Zihan and Zhang, Yuan and Lu, Yan},
    title     = {Single-step Diffusion-based Video Coding with Semantic-Temporal Guidance},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {9752-9761}
}
```


## ⚖️ License

This project as a whole is licensed under [CC BY-NC-SA 4.0](./LICENSE.md).

It incorporates third-party components under their own license terms:

- **DMD2** — CC BY-NC-SA 4.0 (`src/modules/dmd`; covered by `LICENSE.md`)
- **Taming Transformers / VQGAN blocks** — MIT (`src/modules/vqgan`; see `src/modules/vqgan/LICENSE`)
- **DCVC-derived code** — Apache-2.0, MIT, and CC0-1.0 on a file-by-file basis (`src/modules/dcvc.py`, `src/modules/entropy`, and `src/cpp`)
- **FloLPIPS** — MIT (`src/modules/flolpips`; see `src/modules/flolpips/LICENSE`)

See [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) for detailed
attributions, file-level license mapping, and the licenses governing the
Stable Diffusion model weights distributed separately from this repository.
