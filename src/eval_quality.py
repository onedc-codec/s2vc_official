import os, sys, gc
sys.path.append(os.getcwd())   # add the root folder

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

from lpips import LPIPS
from DISTS_pytorch import DISTS
from pytorch_msssim import MS_SSIM
from torchmetrics.image.fid import FrechetInceptionDistance

from modules.blocks.raft import RAFTLargeFlow
from modules.flolpips.flolpips import Flolpips


DEBUG = False


def image_to_255_scale(image, dtype = None):
    """
    Helper function for converting a floating point image to 255 scale.

    The input image is expected to be in the range [0.0, 1.0]. If it is outside
    this range, the function throws an error.

    Args:
        image: A 4-D PyTorch tensor.
        dtype: Output datatype. If not passed, the image is of the same dtype
            as the input.

    Returns:
        The image in [0, 255] scale.
    """
    if image.max() > 1.0:
        raise ValueError("Unexpected image max > 1.0")
    if image.min() < 0.0:
        raise ValueError("Unexpected image min < 0.0")

    image = torch.round(image * 255.0)

    if dtype is not None:
        image = image.to(dtype)

    return image


def update_patch_fid(
    input_images, pred,
    fid_metric = None,
    fid_swav_metric = None,
    patch_size = 256,
    split_patch_num = 2,
):
    """
    Update FID metrics with patch-based calculation.

    This implements the FID/256 method described in the following paper:

    High-Fidelity Generative Image Compression
    Fabian Mentzer, George D. Toderici, Michael Tschannen, Eirikur Agustsson

    First, a user defines a torchmetric class for FID. Then, this function can
    be used to update the metric using the FID/256 calculation.

    This method gives more stable FID calculations for small counts of
    high-resolution images, such as those in the CLIC2020 dataset. Each image
    is divided up into a grid of non-overlapping patches, and the normal FID
    calculation is run treating each patch as an image. Then, the calculation
    is re-run a second time with a patch/2 shift.

    Args:
        input_images: The ground truth images in [0.0, 1.0] range.
        pred: The compressed images in [0.0, 1.0] range.
        fid_metric: A torchmetric for calculationg FID.
        fid_swav_metric: A torchmetric for calculating FID with the SwAV
            backbone.
        patch_size: The patch size to use for dividing up each image.

    Returns:
        The number of patches (metric are updated in-place). The number of
        patches can be used as debugging signal.
    """
    if fid_metric is None and fid_swav_metric is None:
        raise ValueError("At least one metric must not be None.")

    # this applies the FID calculation from Mentzer 2020
    real = image_to_255_scale(
        F.unfold(input_images, kernel_size=patch_size, stride=patch_size)
        .permute(0, 2, 1)
        .reshape(-1, 3, patch_size, patch_size),
        dtype=torch.uint8,
    )
    fake = image_to_255_scale(
        F.unfold(pred, kernel_size=patch_size, stride=patch_size)
        .permute(0, 2, 1)
        .reshape(-1, 3, patch_size, patch_size),
        dtype=torch.uint8,
    )
    patch_count = real.shape[0]
    if fid_metric is not None:
        fid_metric.update(real, real=True)
        fid_metric.update(fake, real=False)
    if fid_swav_metric is not None:
        fid_swav_metric.update(real, real=True)
        fid_swav_metric.update(fake, real=False)

    num_y, num_x = input_images.shape[2], input_images.shape[3]

    unit = patch_size // split_patch_num
    for unit_i in range(1, split_patch_num):
        limit_size = (2. - unit_i / split_patch_num) * patch_size
        if num_y >= limit_size and num_x >= limit_size:
            real = image_to_255_scale(
                F.unfold(
                    input_images[:, :, unit * unit_i:, unit * unit_i:],
                    kernel_size=patch_size,
                    stride=patch_size,
                )
                .permute(0, 2, 1)
                .reshape(-1, 3, patch_size, patch_size),
                dtype=torch.uint8,
            )
            fake = image_to_255_scale(
                F.unfold(
                    pred[:, :, unit * unit_i:, unit * unit_i:],
                    kernel_size=patch_size,
                    stride=patch_size,
                )
                .permute(0, 2, 1)
                .reshape(-1, 3, patch_size, patch_size),
                dtype=torch.uint8,
            )
            patch_count += real.shape[0]
            if fid_metric is not None:
                fid_metric.update(real, real=True)
                fid_metric.update(fake, real=False)
            if fid_swav_metric is not None:
                fid_swav_metric.update(real, real=True)
                fid_swav_metric.update(fake, real=False)

    return patch_count



class video_compare_dataset(Dataset):
    """
    Compare two folders of video frames:
      - label_path:  contains 'im00001.png', 'im00002.png', ...  (strict
                     ``^im(\\d+)\\.png$`` — file names like '0001.png' or
                     'frame_001.png' are rejected.)
      - recon_path:  contains 'im00001_bpp_0.xxxx.png' or 'im00001_0.xxxx.png',
                     ...  (also accepts plain 'im00001.png' without a bpp tag.)
    We match items by the frame index parsed from filenames. The reconstruction
    set may be a subset of the labels, but indices must be continuous from 1..N.
    """

    _re_label = re.compile(r"^im(\d+)\.png$", re.IGNORECASE)
    # supports: im00001_bpp_0.1234.png  OR  im00001_0.1234.png  OR  im00001.png
    _re_recon = re.compile(
        r"^im(\d+)_(?:bpp_)?([0-9]+(?:\.[0-9]+)?)\.png$", re.IGNORECASE
    )
    _re_recon_no_bpp = re.compile(r"^im(\d+)\.png$", re.IGNORECASE)

    def __init__(self, label_path: str, recon_path: str, to_rgb: bool = True):
        super().__init__()
        self.label_dir = Path(label_path)
        self.recon_dir = Path(recon_path)
        if not self.label_dir.is_dir():
            raise NotADirectoryError(f"label_path not found: {self.label_dir}")
        if not self.recon_dir.is_dir():
            raise NotADirectoryError(f"recon_path not found: {self.recon_dir}")

        # 1) Index label frames: index -> filepath
        self._label_index_to_path = {}
        for p in sorted(self.label_dir.glob("*.png")):
            m = self._re_label.match(p.name)
            if m:
                idx = int(m.group(1))
                self._label_index_to_path[idx] = p

        if not self._label_index_to_path:
            raise FileNotFoundError(
                f"No label frames matched pattern 'im{{N:05d}}.png' in {self.label_dir}. "
                f"GT filenames must start with 'im' followed by digits "
                f"(e.g. 'im00001.png'); names like '0001.png' or "
                f"'frame_001.png' are not accepted."
            )

        # 2) Build matched triplets from recon files that also exist in labels
        triplets: List[Tuple[int, Path, Path, float]] = []
        for p in sorted(self.recon_dir.glob("*.png")):
            m = self._re_recon.match(p.name)
            if m:
                idx = int(m.group(1))
                bpp = float(m.group(2))
            else:
                m2 = self._re_recon_no_bpp.match(p.name)
                if not m2:
                    continue
                idx = int(m2.group(1))
                bpp = 0.0
            if idx in self._label_index_to_path:
                triplets.append((idx, self._label_index_to_path[idx], p, bpp))

        if not triplets:
            raise FileNotFoundError(
                f"No matched pairs found. Check filename patterns in {self.recon_dir} "
                "and that indices exist in the label set."
            )

        # 3) Sort by index and enforce continuity 1..N
        triplets.sort(key=lambda x: x[0])
        indices = [t[0] for t in triplets]

        # Sanity: indices must be [1, 2, ..., N] exactly
        N = len(triplets)
        expected = list(range(1, N + 1))
        if indices != expected:
            # Helpful diagnostics
            missing = sorted(set(expected) - set(indices))
            msg = (
                "Reconstruction indices must be continuous from 1..N with no gaps.\n"
                f"  Observed first {min(indices)} last {max(indices)} count {N}\n"
                f"  Expected: {expected[:10]}{'...' if N>10 else ''}\n"
                f"  Label path: {self.label_dir}\n"
                f"  Recon path: {self.recon_dir}\n"
            )
            if missing:
                msg += f"  Missing indices relative to 1..N: {missing[:20]}{'...' if len(missing)>20 else ''}\n"
            msg += (
                "If your recon frames begin at a higher index or have gaps, please re-export them "
                "or rename to a contiguous 1..N set (keeping the original label indices present)."
            )
            raise ValueError(msg)

        self._items = triplets
        self._to_rgb = to_rgb

    def __len__(self):
        return len(self._items)

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        # Convert PIL image to float tensor in [0,1], shape [C, H, W]
        # Robust to L/LA/RGB/RGBA inputs.
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif img.mode == "L":
            # keep single-channel; torch.frombuffer expects RGB? Simpler: convert to tensor via torchvision-like path
            pass
        x = torch.from_numpy(
            (np.array(img, copy=False).astype("float32") / 255.0)
        )
        if x.ndim == 2:           # [H, W] -> [H, W, 1]
            x = x[..., None]
        x = x.permute(2, 0, 1)    # [H, W, C] -> [C, H, W]
        return x

    def __getitem__(self, idx: int):
        # idx is zero-based; stored indices are 1-based, but we already enforced contiguity.
        _, label_p, recon_p, bpp = self._items[idx]

        # Load images
        label_img = Image.open(label_p)
        recon_img = Image.open(recon_p)

        if self._to_rgb:
            label_img = label_img.convert("RGB")
            recon_img = recon_img.convert("RGB")

        # Convert to torch tensors in [0, 1], shape [C, H, W]
        # (Use torchvision.transforms if you prefer; here we keep it dependency-light.)
        import numpy as np  # localized import to avoid global dependency if not used elsewhere

        label = torch.from_numpy((np.array(label_img).astype("float32") / 255.0)).permute(2, 0, 1)
        recon = torch.from_numpy((np.array(recon_img).astype("float32") / 255.0)).permute(2, 0, 1)

        return label, recon, float(bpp)


class Quality_evaluator:
    def __init__(self, label_folder: dict, target_folder: dict,
                 output_path: str, save_pfx="",
                 fid_patch_size=256, fid_patch_num=2,
                 device='cuda:0'):
        """
        label_folder: {'dataset1': {'sequence1': path, ...}, 'dataset2': {...}, ...}
        target_folder: {'dataset1': {'sequence1': path, ...}, 'dataset2': {...}, ...}
        """
        self.device = device
        self.label_folder = label_folder
        self.target_folder = target_folder
        self.sanity_check()

        if DEBUG:
            return

        self.ms_ssim_metric = MS_SSIM(data_range=1.0).to(device).eval()
        self.lpips_metric = LPIPS(net='alex').to(device).eval()
        self.dists_metric = DISTS().to(device).eval()
        self.raft_model = RAFTLargeFlow().to(device).eval()
        self.flolpips_metric = Flolpips().to(device).eval()
        self.flolpips_metric.flownet = self.raft_model    # replace the PWCNet with RAFT

        # this will be initialized during the start of one dataset.
        self.fid_metric = None
        self.fid_patch_size = fid_patch_size
        self.fid_patch_num = fid_patch_num

        # define output path
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.detail_path = self.output_path / "detail"
        self.detail_path.mkdir(parents=True, exist_ok=True)
        self.save_pfx = save_pfx


    def sanity_check(self):
        """ check if the folders are correct
        """
        for dataset in self.label_folder.keys():
            assert dataset in self.target_folder, f"Dataset {dataset} not in target_folder"
            for seq in self.label_folder[dataset].keys():
                assert seq in self.target_folder[dataset], f"Sequence {seq} not in target_folder"


    @torch.no_grad()
    def _forward_flolpips(
            self,
            x0_gt, x1_gt, x2_gt,   # gt t-1, t, t+1  [B,3,H,W] in [0,1]
            x1_hat,               # recon t      [B,3,H,W] in [0,1]
        ) -> torch.Tensor:
        if len(x0_gt.shape) == 3: x0_gt = x0_gt.unsqueeze(0)
        if len(x1_gt.shape) == 3: x1_gt = x1_gt.unsqueeze(0)
        if len(x2_gt.shape) == 3: x2_gt = x2_gt.unsqueeze(0)
        if len(x1_hat.shape) == 3: x1_hat = x1_hat.unsqueeze(0)
        return self.flolpips_metric.forward(
            I0=x0_gt.to(self.device), I1=x2_gt.to(self.device),
            frame_dis=x1_hat.to(self.device), frame_ref=x1_gt.to(self.device),
        )


    @torch.no_grad()
    def evaluate_dataset(self, dataset, seq_dict):
        # 1) Initialize dataset-level metrics
        if not DEBUG:
            self.fid_metric = FrechetInceptionDistance().to(self.device)

        detail_dataset = pd.DataFrame()

        def _as_bchw(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 3:
                x = x.unsqueeze(0)
            return x.to(self.device)

        for idx_seq, seq in enumerate(seq_dict.keys()):
            label_path = self.label_folder[dataset][seq]
            recon_path = self.target_folder[dataset][seq]
            print(f"[{self.device}] Evaluating dataset {dataset}, sequence {seq}")

            ds = video_compare_dataset(label_path, recon_path, to_rgb=True)
            T = len(ds)

            if DEBUG:
                continue

            # Accumulators
            lpips_list_sample, dists_list_sample = [], []
            psnr_list_sample, ms_ssim_list_sample = [], []
            flolpips_list_sample, flolpips_list_sample_valid = [], []
            bpp_list_sample = []

            for i in tqdm(range(T), total=T):
                # Current frame
                x_gt_i, x_hat_i, bpp_i = ds[i]
                x_gt = _as_bchw(x_gt_i)    # [1, C, H, W]
                x_hat = _as_bchw(x_hat_i)  # [1, C, H, W]
                bpp = float(bpp_i if isinstance(bpp_i, (int, float)) else getattr(bpp_i, "item", lambda: float(bpp_i))())
                assert x_gt.shape == x_hat.shape, f"Shape mismatch: GT {tuple(x_gt.shape)} vs Recon {tuple(x_hat.shape)}"
                bpp_list_sample.append(bpp)

                # ----- FLOLPIPS: needs (i-1, i, i+1) in GT and i in recon -----
                if (i >= 1) and (i < T - 1):
                    x0_gt = _as_bchw(ds[i - 1][0])
                    x1_gt = x_gt
                    x2_gt = _as_bchw(ds[i + 1][0])
                    flolpips_val = self._forward_flolpips(
                        x0_gt=x0_gt, x1_gt=x1_gt, x2_gt=x2_gt, x1_hat=x_hat
                    ).mean().item()
                    flolpips_list_sample.append(flolpips_val)
                    flolpips_list_sample_valid.append(flolpips_val)
                else:
                    flolpips_list_sample.append(0.0)

                # ----- Dataset-level FID (patch-wise) -----
                update_patch_fid(
                    x_gt, x_hat,
                    fid_metric=self.fid_metric,
                    patch_size=self.fid_patch_size,
                    split_patch_num=self.fid_patch_num,
                )

                # ----- Frame-wise metrics -----
                mse = torch.mean((x_gt - x_hat) ** 2)
                psnr_list_sample.append((-10.0 * torch.log10(mse.clamp_min(1e-10))).item())
                ms_ssim_list_sample.append(self.ms_ssim_metric(x_gt, x_hat).mean().item())
                lpips_list_sample.append(self.lpips_metric(x_gt, x_hat, normalize=True).mean().item())
                dists_list_sample.append(self.dists_metric(x_gt, x_hat).mean().item())

            # ----- Persist per-frame detail -----
            detail_df = pd.DataFrame({
                'lpips': lpips_list_sample,
                'dists': dists_list_sample,
                'psnr': psnr_list_sample,
                'ms_ssim': ms_ssim_list_sample,
                'flolpips': flolpips_list_sample,
                'bpp': bpp_list_sample,
            })
            detail_name = f"{self.save_pfx}_{seq}_detail.csv" if self.save_pfx else f"{dataset}_{seq}_detail.csv"
            detail_df.to_csv(self.detail_path / detail_name, index=False)

            # ----- Aggregate per-sequence stats -----
            lpips_sample = float(np.mean(lpips_list_sample)) if lpips_list_sample else 0.0
            dists_sample = float(np.mean(dists_list_sample)) if dists_list_sample else 0.0
            psnr_sample = float(np.mean(psnr_list_sample)) if psnr_list_sample else 0.0
            ms_ssim_sample = float(np.mean(ms_ssim_list_sample)) if ms_ssim_list_sample else 0.0
            flolpips_sample = float(np.mean(flolpips_list_sample_valid)) if flolpips_list_sample_valid else 0.0
            bpp_sample = float(np.mean(bpp_list_sample)) if bpp_list_sample else 0.0

            quality_sequence = pd.DataFrame([{
                'name': seq,
                'psnr': psnr_sample,
                'msssim': ms_ssim_sample,
                'lpips': lpips_sample,
                'dists': dists_sample,
                'flolpips': flolpips_sample,
                'bpp': bpp_sample,
            }], index=[idx_seq])
            detail_dataset = pd.concat([detail_dataset, quality_sequence])

        if DEBUG:
            return

        # 3) Dataset-level metric
        fid_value = self.fid_metric.compute().item()

        summary_dataset = detail_dataset.mean(numeric_only=True).to_dict()
        summary_dataset.update({
            'name': 'summary',
            'fid': fid_value,
        })
        summary_df = pd.DataFrame([summary_dataset])

        # 4) Save
        detail_dataset_name = f"{self.save_pfx}_{dataset}_detail.csv" if self.save_pfx else f"{dataset}_detail.csv"
        detail_dataset.to_csv(self.output_path / detail_dataset_name, index=False)
        print(f"Saved detail results to {self.output_path / detail_dataset_name}")

        summary_df_name = f"{self.save_pfx}_{dataset}_summary.csv" if self.save_pfx else f"{dataset}_summary.csv"
        summary_df.to_csv(self.output_path / summary_df_name, index=False)
        print(f"Saved summary results to {self.output_path / summary_df_name}")


    def run(self):
        for dataset, seq_dict in self.label_folder.items():
            self.evaluate_dataset(dataset, seq_dict)
        gc.collect()
        torch.cuda.empty_cache()


def discover_pairs(label_root: Path, recon_root_for_ds: Path, qp_name: str):
    """Walk ``label_root`` for per-sequence subdirectories, pair each one with
    its reconstruction folder ``recon_root_for_ds/<seq>/<qp_name>/``, and
    return two dicts keyed by sequence name.

    Sequences whose label dir contains no ``im{N:05d}.png`` files, or whose
    expected recon dir does not exist, are skipped with a warning.
    """
    label_map, recon_map = {}, {}
    re_label = re.compile(r"^im\d+\.png$", re.IGNORECASE)

    for seq_dir in sorted(label_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq = seq_dir.name

        # check the label dir actually contains im{N:05d}.png frames
        has_im = any(re_label.match(f.name) for f in seq_dir.iterdir() if f.is_file())
        if not has_im:
            print(f"  [skip] {seq}: no 'im{{N:05d}}.png' frames in {seq_dir}")
            continue

        recon_dir = recon_root_for_ds / seq / qp_name
        if not recon_dir.is_dir():
            print(f"  [skip] {seq}: recon dir not found at {recon_dir}")
            continue

        label_map[seq] = str(seq_dir)
        recon_map[seq] = str(recon_dir)

    return label_map, recon_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate reconstructed videos against ground truth (PSNR, "
                    "MS-SSIM, LPIPS, DISTS, FloLPIPS, bpp, patch-FID). "
                    "Designed to consume inference.py's output directly.",
    )
    parser.add_argument("--label_root", required=True,
        help="Root directory containing per-sequence GT subdirectories with "
             "im{N:05d}.png files. Same as inference.py's --video_ds_folder.")
    parser.add_argument("--recon_base_path", required=True,
        help="Inference's --output_base_path. Reconstructions for each "
             "sequence are read from <recon_base_path>/<ds_name>/<seq>/<qp_name>/.")
    parser.add_argument("--ds_name", required=True,
        help="Dataset tag (e.g. UVG); must match the --ds_name passed to inference.py.")
    parser.add_argument("--qp_name", required=True,
        help="Rate-point tag (e.g. lmbda1.8); must match the --qp_name passed to inference.py.")
    parser.add_argument("--output_path", required=True,
        help="Directory to write the resulting CSV files.")
    parser.add_argument("--save_pfx", default="",
        help="Optional prefix prepended to output CSV filenames.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fid_patch_size", type=int, default=256)
    parser.add_argument("--fid_patch_num", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    label_root = Path(args.label_root)
    recon_root_for_ds = Path(args.recon_base_path) / args.ds_name

    print(f"[INFO] dataset       : {args.ds_name}")
    print(f"[INFO] rate point    : {args.qp_name}")
    print(f"[INFO] label root    : {label_root}")
    print(f"[INFO] recon root    : {recon_root_for_ds}/*/{args.qp_name}")
    print(f"[INFO] output path   : {args.output_path}")
    print(f"[INFO] scanning sequences...")

    label_map, recon_map = discover_pairs(label_root, recon_root_for_ds, args.qp_name)

    if not label_map:
        raise RuntimeError(
            f"No valid (label, recon) pairs found.\n"
            f"  label root: {label_root}\n"
            f"  recon root: {recon_root_for_ds}/<seq>/{args.qp_name}\n"
            f"Check that both --label_root and --recon_base_path / --ds_name / "
            f"--qp_name are correct, and that GT filenames follow 'im{{N:05d}}.png'."
        )

    print(f"[INFO] paired {len(label_map)} sequence(s): {list(label_map.keys())}")

    label_folder  = {args.ds_name: label_map}
    target_folder = {args.ds_name: recon_map}

    evaluator = Quality_evaluator(
        label_folder, target_folder, args.output_path,
        save_pfx=args.save_pfx,
        fid_patch_size=args.fid_patch_size,
        fid_patch_num=args.fid_patch_num,
        device=args.device,
    )
    evaluator.run()
