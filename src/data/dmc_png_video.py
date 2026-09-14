import os, re, glob
from typing import List, Iterable, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# ------------ utilities ------------
_num_re = re.compile(r"(\d+)")
def _natkey(s: str):
    b = os.path.basename(s)
    return [int(t) if t.isdigit() else t.lower() for t in _num_re.split(b)]

def _list_pngs(seq_dir: str) -> List[str]:
    files = glob.glob(os.path.join(seq_dir, "*.png")) + glob.glob(os.path.join(seq_dir, "*.PNG"))
    return sorted(files, key=_natkey)

def scan_dataset(root_dir: str) -> List[str]:
    seqs = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and any(f.lower().endswith(".png") for f in os.listdir(p)):
            seqs.append(p)
    return seqs

def count_frames_per_sequence(seq_dirs: Iterable[str]) -> List[Tuple[str, int]]:
    return [(d, len(_list_pngs(d))) for d in seq_dirs]

# --------- minimal PNG reader (PIL) ---------
class PNGSequenceReader:
    def __init__(self, seq_dir: str, device: str = "cpu"):
        self.seq_dir = seq_dir
        self.device = device
        self.frame_files = _list_pngs(seq_dir)
        if not self.frame_files:
            raise FileNotFoundError(f"No PNG frames in {seq_dir}")
        self.num_frames = len(self.frame_files)

    def __len__(self): return self.num_frames

    def _read_rgb_tensor(self, path: str) -> torch.Tensor:
        # returns uint8 tensor [C,H,W]
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.array(im, copy=False)              # H,W,3 uint8
        t = torch.from_numpy(arr)                       # H,W,3 uint8
        t = t.permute(2, 0, 1).contiguous()             # 3,H,W
        return t

    def __getitem__(self, idx_or_slice) -> torch.Tensor:
        if isinstance(idx_or_slice, int):
            idxs = [idx_or_slice]
        elif isinstance(idx_or_slice, slice):
            start = idx_or_slice.start or 0
            stop  = idx_or_slice.stop  or len(self)
            idxs = range(max(0, start), min(stop, len(self)))
        else:
            raise TypeError("Index must be int or slice.")
        frames = [self._read_rgb_tensor(self.frame_files[i]) for i in idxs]
        out = torch.stack(frames, 0) if frames else torch.empty(0, 3, 0, 0, dtype=torch.uint8)
        return out.to(self.device, non_blocking=True) if self.device != "cpu" else out

# --------------- dataset ---------------
class VideoDataset_Eval(Dataset):
    """
    `video_paths`: list of sequence dirs, or a text file, or a root dir to scan.
    Returns:
      - 'video': float32 in [0,1], [T,C,H,W], T<=num_frames
      - 'file_name': sequence folder name
      - 'num_frames_total': total frames in sequence
    """
    def __init__(self, video_paths: Union[List[str], str], num_frames: int = 16, device: str = "cpu"):
        if isinstance(video_paths, list):
            seqs = video_paths
        else:
            if os.path.isdir(video_paths):
                seqs = scan_dataset(video_paths)
            else:
                with open(video_paths, "r") as f:
                    seqs = [ln.strip() for ln in f if ln.strip()]
        if not seqs:
            raise ValueError("No valid sequence directories found.")
        self.video_paths = seqs
        self.num_frames  = int(num_frames)
        self.device      = device
        self._counts     = [len(_list_pngs(p)) for p in self.video_paths]

    def __len__(self): return len(self.video_paths)

    def __getitem__(self, idx):
        seq_dir   = self.video_paths[idx]
        file_name = os.path.basename(seq_dir)
        reader    = PNGSequenceReader(seq_dir, device=self.device)
        T         = min(self.num_frames, len(reader))
        frames    = reader[:T].float() / 255.0
        return {"video": frames, "file_name": file_name, "num_frames_total": len(reader)}

    def frame_count(self, idx: int) -> int:
        return self._counts[idx]


# -------------- example --------------
if __name__ == "__main__":
    # seqs = scan_dataset(root)
    # print(count_frames_per_sequence(seqs))
    # ds = VideoDataset_Eval(root, num_frames=16, device="cpu")
    # print(ds[0]["file_name"], ds[0]["video"].shape, ds[0]["num_frames_total"])
    pass
