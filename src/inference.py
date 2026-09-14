import argparse
import os
import shutil
import sys
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from torchvision.utils import save_image
from diffusers.utils import export_to_video

from data.dmc_png_video import VideoDataset_Eval
from models.codec import OneDCVideoCodec


def im_name(dir, idx, bpp):
    """Saved frame index is 1-based; in code it is 0-based."""
    return os.path.join(dir, f"im{idx + 1:05d}_bpp_{bpp:.6f}.png")


class Evaluation:
    def __init__(self, args):
        self.args = args

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # output paths
        self.qp_name = args.qp_name
        self.ds_name = args.ds_name
        self.output_base_path = os.path.join(args.output_base_path, self.ds_name)
        self.mp4_output_path = os.path.join(self.output_base_path, "recon_mp4")

        os.makedirs(self.output_base_path, exist_ok=True)
        os.makedirs(self.mp4_output_path, exist_ok=True)

        # plumb CLI paths into the args namespace expected by the model
        args.i_codec_ckpt = args.i_ckpt_path
        args.unet_ckpt_lora = os.path.join(args.ckpt_path, "model.safetensors")
        args.p_codec_ckpt = os.path.join(args.ckpt_path, "model_1.safetensors")

        # build model
        self.model = OneDCVideoCodec(args, self.device).to(self.device)
        self.model.eval()
        self.model.p_frame_model.update()

        # dataset
        eval_dataset = VideoDataset_Eval(args.video_ds_folder, num_frames=args.n_frame)
        self.eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, num_workers=1,
            batch_size=1, shuffle=False,
            drop_last=False,
            prefetch_factor=None,
        )

    @torch.no_grad()
    def evaluate(self):
        total_seq = len(self.eval_dataloader)
        for idx, batch in enumerate(self.eval_dataloader):
            video = batch['video'].to(self.device)
            video = video * 2 - 1                       # to [-1, 1]
            file_name = batch['file_name'][0].split('.')[0]
            print(f"Processing video {idx + 1}/{total_seq}: {file_name}")

            png_output_dir = os.path.join(self.output_base_path, file_name, self.qp_name)
            if os.path.exists(png_output_dir):
                shutil.rmtree(png_output_dir)
            os.makedirs(png_output_dir, exist_ok=True)

            # 1. pad video to multiple of 64
            B, T, C, H, W = video.shape
            pad_H = (64 - H % 64) % 64
            pad_W = (64 - W % 64) % 64
            video_pad = torch.nn.functional.pad(
                video.squeeze(0), (0, pad_W, 0, pad_H), mode='reflect'
            ).unsqueeze(0)
            Hp, Wp = video_pad.shape[3], video_pad.shape[4]

            # 2. encode I-frame
            x0 = video_pad[:, 0]
            i_enc_dict = self.model.forward_i_frame(x0, recon_to_img=True)
            init_dpb = i_enc_dict['dpb']
            bpp_list = [i_enc_dict['i_bpp'].item()]
            recon_list = [i_enc_dict['i_frame_recon_img']]

            # 2.1 save I-frame recon
            frame_idx = 0
            im_path = im_name(png_output_dir, frame_idx, bpp_list[frame_idx])
            im_recon = deepcopy(i_enc_dict['i_frame_recon_img'])
            im_recon = im_recon[:, :, :H, :W].clamp(-1, 1) * 0.5 + 0.5
            save_image(im_recon, im_path)

            # 3. encode P-frames
            bit_stream_list = []
            dpb = deepcopy(init_dpb)
            for frame_idx in tqdm(range(1, T), desc="Encoding P frames"):
                is_reset = ((frame_idx - 1) % self.args.n_reset == 0)
                x = video_pad[:, frame_idx]
                p_enc_dict = self.model.encode_one_frame(x, dpb, is_reset)
                bit_stream_list.append(p_enc_dict['bit_stream'])
                bpp_list.append(len(p_enc_dict['bit_stream']) * 8 / (H * W))
                dpb = p_enc_dict['dpb']

            # 4. decode P-frames
            dpb = deepcopy(init_dpb)
            for frame_idx in tqdm(range(1, T), desc="Decoding P frames"):
                is_reset = ((frame_idx - 1) % self.args.n_reset == 0)
                p_dec_dict = self.model.decode_one_frame(
                    bit_stream_list[frame_idx - 1], dpb, Hp, Wp, is_reset
                )
                recon_list.append(p_dec_dict['x_hat'])
                dpb = p_dec_dict['dpb']

                im_path = im_name(png_output_dir, frame_idx, bpp_list[frame_idx])
                im_recon = deepcopy(p_dec_dict['x_hat'])
                im_recon = im_recon[:, :, :H, :W].clamp(-1, 1) * 0.5 + 0.5
                save_image(im_recon, im_path)

            # 5. save mp4
            recon_video = torch.stack(recon_list, dim=1).clamp(-1, 1) * 0.5 + 0.5
            recon_video = recon_video[:, :, :, :H, :W]
            mp4_output_dir = os.path.join(self.mp4_output_path, self.qp_name)
            os.makedirs(mp4_output_dir, exist_ok=True)
            mp4_output_file = os.path.join(mp4_output_dir, f"{file_name}.mp4")
            recon_video = recon_video[0].float().permute(0, 2, 3, 1).cpu().numpy()
            export_to_video(recon_video, mp4_output_file)

            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_base_path", type=str, required=True)
    parser.add_argument("--i_ckpt_path", type=str, required=True,
                        help="Path to I-frame codec safetensors.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Folder containing model.safetensors (UNet+LoRA) "
                             "and model_1.safetensors (P-frame codec).")
    parser.add_argument("--ds_name", type=str, required=True, help="Dataset name (e.g. UVG, MCL-JCV).")
    parser.add_argument("--qp_name", type=str, required=True, help="QP tag (e.g. qp0, qp1).")
    parser.add_argument("--video_ds_folder", type=str, required=True,
                        help="Folder of per-sequence PNG dirs, or a txt file listing them.")
    parser.add_argument("--n_frame", type=int, default=96)
    parser.add_argument("--n_reset", type=int, default=32)
    args = parser.parse_args()

    args_conf = OmegaConf.create(vars(args))
    config = OmegaConf.load(args.config_path)
    return OmegaConf.merge(config, args_conf)


if __name__ == "__main__":
    args = parse_args()
    evaluator = Evaluation(args)
    evaluator.evaluate()
