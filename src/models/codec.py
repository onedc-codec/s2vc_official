import omegaconf
import torch
from torch import nn
from einops import rearrange
from diffusers import DDIMScheduler

from utils import load_safetensor
from modules.vae.autoencoders_patch_attn import AutoencoderKL_patch_attn
from modules.dmd.utils import get_x0_from_noise, DummyModule
from models.codec_module import IntraNoAR
from models.video_codec import DMC_OneDC
from models.decoder_unet import prepare_unet_for_codec


# Default HuggingFace Hub IDs the original training pipeline used.
# Override these per-run via the `sd15` / `sd21_vae` fields in the YAML config
# (useful when the upstream HuggingFace repo is unavailable; point to a local
# snapshot dir with the standard diffusers layout instead).
SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SD21_VAE_MODEL_ID = "stabilityai/stable-diffusion-2-1"


class OneDCVideoCodec(nn.Module):
    """One-step diffusion video codec (inference only).

    Stages of a single forward:
        I-frame  -> ``forward_i_frame(image)``                       -> dpb
        P-frame  -> ``encode_one_frame(image, dpb, is_reset)``       -> bit_stream, dpb
        P-frame  -> ``decode_one_frame(stream, dpb, H, W, is_reset)``-> recon, dpb

    The ``dpb`` (decoded picture buffer) is a small dict carried between
    frames; see :func:`forward_i_frame` for the layout.
    """

    def __init__(self, args: omegaconf.OmegaConf, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.vae_dim = 4
        self.vae_attn_patch = args.vae_attn_patch

        # Resolve pretrained-model paths (local override > HF Hub default)
        sd15_path     = getattr(args, "sd15", None) or SD15_MODEL_ID
        sd21_vae_path = getattr(args, "sd21_vae", None) or SD21_VAE_MODEL_ID
        print(f"[INFO] using SD 1.5 from: {sd15_path}")
        print(f"[INFO] using SD 2.1 VAE from: {sd21_vae_path}")

        # ----- VAE (SD 2.1 large VAE with windowed mid-block attention) -----
        # The tiny TAESD path used during early training is replaced by a
        # no-op stub since inference always uses the large VAE.
        self.vae = DummyModule().float().to(device)
        self.vae_large = AutoencoderKL_patch_attn.from_pretrained(
            sd21_vae_path, subfolder="vae", torch_dtype=torch.float32
        ).float().to(device)
        self.vae_large.requires_grad_(False)
        self.vae_large.set_attn_patch(self.vae_attn_patch)

        # ----- one-step diffusion UNet -----
        self.ch_diffusion = 320
        self.ch_semantic = 768
        self.feedforward_model = prepare_unet_for_codec(
            in_ch=self.ch_diffusion,
            motion_ch=args.p_codec.motion_ch,
            lora_config=args.lora_config,
            sd15_model_id=sd15_path,
        ).to(device)
        self.feedforward_model.requires_grad_(False)

        # ----- I-frame codec -----
        self.i_frame_model = IntraNoAR(
            cond_ch=self.vae_dim,
            ctrl_ch=self.ch_diffusion,
            internal_ch=args.i_codec.internal_ch,
            bottleneck_ch=args.i_codec.bottleneck_ch,
            unet_ch_config=tuple(args.i_codec.unet_ch_config),
            z_fsq_levels=list(args.i_codec.z_fsq_levels),
        ).to(device)
        self.i_frame_model.requires_grad_(False)

        # ----- P-frame codec -----
        self.p_frame_model = DMC_OneDC(
            ch=args.p_codec.internal_ch,
            ch_y=args.p_codec.bottleneck_ch,
            ch_z=args.p_codec.hyperbottleneck_ch,
            ch_diffusion=self.ch_diffusion,
            ch_semantic=self.ch_semantic,
            ch_motion=args.p_codec.motion_ch,
            ch_vae=self.vae_dim,
            detach_motion_feat=True,
        ).to(device)
        self.p_frame_model.requires_grad_(False)
        self.p_frame_model.set_use_ckpt(False)
        self.dpb = self.p_frame_model.dpb
        self.remove_time_dim = self.p_frame_model.remove_time_dim
        self.add_time_dim = self.p_frame_model.add_time_dim

        # ----- load checkpoints -----
        self._load_ckpt()

        # ----- DDIM scheduler (only alphas_cumprod is read) -----
        self.scheduler = DDIMScheduler.from_pretrained(sd15_path, subfolder="scheduler")
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        self.conditioning_timestep = args.conditioning_timestep

        torch.cuda.empty_cache()

    def _load_ckpt(self):
        """Load the three trained checkpoints. All three are required."""
        args = self.args

        assert args.i_codec_ckpt is not None, "args.i_codec_ckpt is required"
        print(f"[INFO] loading I-frame codec from {args.i_codec_ckpt}")
        sd = load_safetensor(args.i_codec_ckpt, map_location="cpu")
        msg = self.i_frame_model.load_state_dict(sd, strict=True)
        print(msg)

        assert args.p_codec_ckpt is not None, "args.p_codec_ckpt is required"
        print(f"[INFO] loading P-frame codec from {args.p_codec_ckpt}")
        sd = load_safetensor(args.p_codec_ckpt, map_location="cpu")
        msg = self.p_frame_model.load_state_dict(sd, strict=False)
        print(msg)

        assert args.unet_ckpt_lora is not None, "args.unet_ckpt_lora is required"
        print(f"[INFO] loading UNet + LoRA from {args.unet_ckpt_lora}")
        sd = load_safetensor(args.unet_ckpt_lora, map_location="cpu")
        msg = self.feedforward_model.load_state_dict(sd, strict=False)
        print(msg)

    @torch.no_grad()
    def vae_encode_image(self, image):
        latents = self.vae_large.encode(image).latent_dist.sample()
        latents = self.vae_large.config.scaling_factor * latents
        return latents.float().detach()

    def vae_decode_image(self, latents):
        latents = 1 / self.vae_large.config.scaling_factor * latents
        image = self.vae_large.decode(latents).sample.float()
        return image

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward_i_frame(self, image, recon_to_img=False):
        """Encode the first frame as an I-frame.

        Args:
            image: ``[B, 3, H, W]`` in ``[-1, 1]``.
            recon_to_img: if True, also return the pixel-space reconstruction.

        Returns dict with keys::

            i_bpp                 bpp under hard quantisation
            i_frame_cond          UNet input feature
            i_frame_recon         x0 latent estimate, [B, 4, H/8, W/8]
            i_frame_recon_img     pixel recon (if recon_to_img), [B, 3, H, W]
            dpb                   {ref_feature, ref_frame, ref_unet_feature}
        """
        device = self.device
        x_pix = image
        x_latent = self.vae_encode_image(image)
        timesteps = torch.ones(x_pix.shape[0], device=device, dtype=torch.long) * self.conditioning_timestep

        # 1. IntraNoAR encodes the I-frame and produces the UNet condition
        self.feedforward_model.set_adapters('iframe')
        enc_dict = self.i_frame_model(x=x_pix, cond=x_latent, fix_codec=True, fix_encoder=True)
        unet_input = enc_dict['x_hat'].detach()
        unet_semantic = rearrange(enc_dict['y_semantic'].detach(),
                                  'b c h w -> b (h w) c').contiguous()

        # 2. SD1.5 UNet with iframe LoRA: 1-step denoise to obtain x0
        sample, dummy_noise, prev_feats = self.feedforward_model(
            sample=unet_input,
            timestep=timesteps.long(),
            encoder_hidden_states=unet_semantic,
            added_cond_kwargs=None,
        )
        student_x0_pred = get_x0_from_noise(
            dummy_noise.double(), sample.double(), self.alphas_cumprod.double(), timesteps
        ).float()

        pred_image = self.vae_decode_image(student_x0_pred).detach() if recon_to_img else None

        # switch back so subsequent P-frame forwards see the right adapter
        self.feedforward_model.set_adapters('pframe')

        return {
            'i_bpp': enc_dict['bpp_hard_y'],
            'i_frame_cond': unet_input,
            'i_frame_recon': student_x0_pred,
            'i_frame_recon_img': pred_image,
            'dpb': {
                'ref_feature': unet_input.detach(),
                'ref_frame': student_x0_pred.detach(),
                'ref_unet_feature': [f.detach() for f in prev_feats],
            },
        }

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float32)        # MUST be float32 to avoid mismatch
    def encode_one_frame(self, image, dpb, is_reset):
        """Compress one P-frame to a bit stream.

        Args:
            image: ``[1, 3, H, W]`` in ``[-1, 1]`` (batch must be 1).
            dpb: decoded picture buffer from previous frame.
            is_reset: True for the first P-frame after an I-frame / reset.

        Returns dict::

            bit_stream    bytes
            dpb           updated buffer (ref_feature/ref_frame/ref_unet_feature)
            f_pix         pre-UNet pixel feature (debugging)
            f_semantic    semantic feature fed as encoder_hidden_states
        """
        device = self.device

        # 1. P-frame codec: produce bit stream + pre-UNet features
        enc_dict = self.p_frame_model.compress(image, dpb, is_reset)
        sample = enc_dict['f_pix']
        unet_semantic = rearrange(enc_dict['f_semantic'], 'b c h w -> b (h w) c').contiguous()
        timesteps = torch.ones(1, device=device, dtype=torch.long) * self.conditioning_timestep

        # 2. P-frame UNet: 1-step denoise + collect per-block features
        self.feedforward_model.set_adapters('pframe')
        sample, dummy_noise, prev_feats = self.feedforward_model(
            sample=sample,
            timestep=timesteps.long(),
            encoder_hidden_states=unet_semantic,
            added_cond_kwargs=None,
            prev_feats=dpb['ref_unet_feature'],
        )
        student_x0_pred = get_x0_from_noise(
            dummy_noise.double(), sample.double(), self.alphas_cumprod.double(), timesteps
        ).float()
        pred_image = self.vae_decode_image(student_x0_pred)

        # update dpb
        enc_dict['dpb']['ref_feature'] = enc_dict['dpb']['ref_feature'].detach()
        enc_dict['dpb']['ref_frame'] = student_x0_pred.detach()
        enc_dict['dpb']['ref_unet_feature'] = [f.detach() for f in prev_feats]
        enc_dict['x_hat'] = pred_image.detach()
        enc_dict['x'] = image
        return enc_dict

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float32)        # MUST be float32 to avoid mismatch
    def decode_one_frame(self, bit_stream, dpb, height, width, is_reset):
        """Decode one P-frame from a bit stream.

        Args:
            bit_stream: bytes produced by :func:`encode_one_frame`.
            dpb: decoded picture buffer from previous frame.
            height, width: padded frame size (multiple of 64).
            is_reset: True for the first P-frame after an I-frame / reset.

        Returns dict::

            x_hat   pixel-space reconstruction, [1, 3, H, W] in [-1, 1]
            dpb     updated buffer
            f_pix, f_semantic   intermediate features
        """
        device = self.device

        # 1. decompress to obtain pre-UNet features
        dec_dict = self.p_frame_model.decompress(bit_stream, dpb, height, width, is_reset)
        sample = dec_dict['f_pix']
        unet_semantic = rearrange(dec_dict['f_semantic'], 'b c h w -> b (h w) c').contiguous()
        timesteps = torch.ones(1, device=device, dtype=torch.long) * self.conditioning_timestep

        # 2. P-frame UNet 1-step denoise
        self.feedforward_model.set_adapters('pframe')
        sample, dummy_noise, prev_feats = self.feedforward_model(
            sample=sample,
            timestep=timesteps.long(),
            encoder_hidden_states=unet_semantic,
            added_cond_kwargs=None,
            prev_feats=dpb['ref_unet_feature'],
        )
        student_x0_pred = get_x0_from_noise(
            dummy_noise.double(), sample.double(), self.alphas_cumprod.double(), timesteps
        ).float()
        pred_image = self.vae_decode_image(student_x0_pred)

        dec_dict['dpb']['ref_feature'] = dec_dict['dpb']['ref_feature'].detach()
        dec_dict['dpb']['ref_frame'] = student_x0_pred.detach()
        dec_dict['dpb']['ref_unet_feature'] = [f.detach() for f in prev_feats]
        dec_dict['x_hat'] = pred_image.detach()
        return dec_dict
