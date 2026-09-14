import torch
from torch import nn

from modules.dcvc import DepthConvBlock4, DepthConvBlock5, ConvFFN4, \
                         ResidualBlockWithStride, ResidualBlockUpsample
from modules.blocks.conv import ResBlock
from modules.vqgan.blocks import AttnBlock
from modules.entropy.compression_model_video import CompressionModel
from modules.entropy.cuda_inference import round_and_to_int8


class CkptModule(nn.Module):
    """Submodule whose forward can optionally be wrapped in gradient
    checkpointing. At inference (no autograd) this is a no-op, so we keep the
    structure but default ``use_ckpt=False``.
    """
    def __init__(self):
        super().__init__()
        self.use_ckpt = False

    def set_use_ckpt(self, use_ckpt=True):
        self.use_ckpt = use_ckpt

    def internal_forward(self, *args):
        raise NotImplementedError

    def forward(self, *args):
        return self.internal_forward(*args)


class FeatureExtractor(CkptModule):
    """Extract temporal feature from previous frame.

    Output:
        ctx: for current frame encoding
        ctx_t: for temporal prior in entropy model
    """
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthConvBlock5(ch, ch),
            DepthConvBlock5(ch, ch),
            DepthConvBlock5(ch, ch),
        )
        self.conv2 = nn.Sequential(
            DepthConvBlock5(ch, ch),
            DepthConvBlock5(ch, ch),
            DepthConvBlock5(ch, ch),
        )

    def internal_forward(self, x):
        ctx = self.conv1(x)
        ctx_t = self.conv2(ctx)
        return ctx, ctx_t


class Encoder(CkptModule):
    """Conditional encoder with previous frame feature."""
    def __init__(self, ch, ch_y):
        super().__init__()
        ch0 = ch * 3 // 2
        ch1 = ch * 2

        self.conv1 = nn.Sequential(
            DepthConvBlock5(ch1, ch0),
            DepthConvBlock5(ch0, ch0),
            DepthConvBlock5(ch0, ch0),
            DepthConvBlock5(ch0, ch),
            DepthConvBlock5(ch, ch),
            DepthConvBlock5(ch, ch),
        )
        self.conv2 = nn.Sequential(
            ResidualBlockWithStride(ch, ch, stride=2),
            ResBlock(ch, ch),
            nn.Conv2d(ch, ch_y, kernel_size=1),
        )

    def internal_forward(self, x, ctx):
        x = torch.cat((x, ctx), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder(CkptModule):
    """Conditional decoder with previous frame feature."""
    def __init__(self, ch, ch_y):
        super().__init__()
        ch0 = ch * 3 // 2
        ch1 = ch * 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_y, ch, kernel_size=1),
            ResBlock(ch, ch),
            ResidualBlockUpsample(ch, ch),
        )
        self.conv2 = nn.Sequential(
            DepthConvBlock5(ch1, ch0),
            DepthConvBlock5(ch0, ch0),
            DepthConvBlock5(ch0, ch0),
            DepthConvBlock5(ch0, ch),
            DepthConvBlock5(ch, ch),
            DepthConvBlock5(ch, ch),
        )

    def internal_forward(self, x, ctx):
        x = self.conv1(x)
        x = torch.cat((x, ctx), dim=1)
        x = self.conv2(x)
        return x


class ReconHead(CkptModule):
    """Map decoder feature f_t to one-step diffusion recon."""
    def __init__(self, ch, ch_diffusion):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(ch, ch),
            ResBlock(ch, ch),
            ResBlock(ch, ch_diffusion),
        )

    def internal_forward(self, x):
        return self.net(x)


class SemanticHead(CkptModule):
    """Map decoder feature f_t to semantic feature c to guide diffusion."""
    def __init__(self, ch, ch_semantic):
        super().__init__()
        ch0 = ch * 3 // 2
        self.net_ds = nn.Sequential(
            ResidualBlockWithStride(ch, ch0, stride=2),
            ResidualBlockWithStride(ch0, ch0, stride=2),
            ResBlock(ch0, ch0),
        )
        self.net_attn = nn.Sequential(
            nn.Conv2d(ch0, ch_semantic, kernel_size=2, stride=2),
            AttnBlock(ch_semantic),
            ConvFFN4(ch_semantic),
            AttnBlock(ch_semantic),
            ConvFFN4(ch_semantic),
        )

    def internal_forward(self, x):
        x = self.net_ds(x)
        x = self.net_attn(x)
        return x


class HyperEncoder(CkptModule):
    def __init__(self, ch_y, ch_z):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock4(ch_y, ch_y),
            DepthConvBlock4(ch_y, ch_y),
            ResidualBlockWithStride(ch_y, ch_y),
            ResidualBlockWithStride(ch_y, ch_y),
            nn.Conv2d(ch_y, ch_z, kernel_size=1),
        )

    def internal_forward(self, x):
        return self.conv(x)


class HyperDecoder(CkptModule):
    def __init__(self, ch_y, ch_z):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_z, ch_y, kernel_size=1),
            ResidualBlockUpsample(ch_y, ch_y),
            ResidualBlockUpsample(ch_y, ch_y),
            DepthConvBlock4(ch_y, ch_y),
            DepthConvBlock4(ch_y, ch_y),
        )

    def internal_forward(self, x):
        return self.conv(x)


class TemporalPrior(CkptModule):
    def __init__(self, ch, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockWithStride(ch, ch_out),
            DepthConvBlock4(ch_out, ch_out),
        )

    def internal_forward(self, x):
        return self.conv(x)


class PriorFusion(CkptModule):
    def __init__(self, ch_y):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock4(ch_y * 3, ch_y * 3),
            DepthConvBlock4(ch_y * 3, ch_y * 3),
            DepthConvBlock4(ch_y * 3, ch_y * 3),
        )

    def internal_forward(self, x):
        return self.conv(x)


class SpatialPrior(CkptModule):
    def __init__(self, ch_y):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock4(ch_y * 4, ch_y * 3),
            DepthConvBlock4(ch_y * 3, ch_y * 3),
            DepthConvBlock4(ch_y * 3, ch_y * 2),
        )

    def internal_forward(self, x):
        return self.conv(x)


class DMC_OneDC(CompressionModel):
    """P-frame codec. Inference-only public surface: ``compress`` /
    ``decompress`` write / read the bit stream, plus helper hooks called from
    the outer codec (``apply_feature_adaptor`` / ``remove_time_dim`` /
    ``add_time_dim``). The training-only ``_forward`` paths have been removed.
    """
    def __init__(self, ch, ch_y, ch_z, ch_diffusion, ch_semantic, ch_motion,
                 ch_vae=4, z_lmbda_scale=0.9, detach_motion_feat=True):
        super().__init__(z_channel=ch_z, qp_num=1)

        self.pix_emb = nn.Conv2d(3, ch, kernel_size=8, stride=8)
        self.feature_adaptor_i = ResBlock(ch_vae, ch)
        self.feature_adaptor_p = ResBlock(ch, ch)
        self.feature_extractor = FeatureExtractor(ch)

        self.encoder = Encoder(ch, ch_y)

        self.hyper_encoder = HyperEncoder(ch_y, ch_z)
        self.hyper_decoder = HyperDecoder(ch_y, ch_z)

        self.temporal_prior_encoder = TemporalPrior(ch, ch_y * 2)
        self.y_prior_fusion = PriorFusion(ch_y)
        self.y_spatial_prior = SpatialPrior(ch_y)

        self.decoder = Decoder(ch, ch_y)
        self.recon_pix = ReconHead(ch, ch_diffusion)
        self.recon_semantic = SemanticHead(ch, ch_semantic)

        self.dpb = {
            'ref_frame': None,
            'ref_feature': None,
        }
        self.frame_idx = 1
        self.z_lmbda_scale = z_lmbda_scale

        self.set_use_ckpt(False)        # inference defaults to no checkpointing

    def set_use_ckpt(self, use_ckpt=True):
        for m in self.modules():
            if isinstance(m, CkptModule):
                m.set_use_ckpt(use_ckpt)

    @staticmethod
    def remove_time_dim(x):
        """[B, 1, C, H, W] -> [B, C, H, W]"""
        if len(x.shape) == 5:
            assert x.shape[1] == 1
            x = x.squeeze(1)
        return x

    @staticmethod
    def add_time_dim(x):
        """[B, C, H, W] -> [B, 1, C, H, W]"""
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        return x

    def apply_feature_adaptor(self, dpb, is_reset=False):
        if is_reset:
            assert dpb["ref_frame"] is not None, "I-frame latent missing for reset."
            dpb["ref_frame"] = self.remove_time_dim(dpb["ref_frame"])
            return self.feature_adaptor_i(dpb["ref_frame"])
        else:
            assert dpb["ref_feature"] is not None, "P-frame latent missing for non-reset."
            dpb["ref_feature"] = self.remove_time_dim(dpb["ref_feature"])
            return self.feature_adaptor_p(dpb["ref_feature"])

    def res_prior_param_decoder(self, z_hat, ctx_t):
        hierarchical_params = self.hyper_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(ctx_t)
        _, _, H, W = temporal_params.shape
        hierarchical_params = hierarchical_params[:, :, :H, :W].contiguous()
        params = self.y_prior_fusion(torch.cat((hierarchical_params, temporal_params), dim=1))
        return params

    def get_recon_and_feature(self, y_hat, ctx, f_prev):
        feature = self.decoder(y_hat, ctx)
        f_pix = self.recon_pix(feature)
        f_semantic = self.recon_semantic(feature)
        return feature, f_pix, f_semantic

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float32)        # must be float32 to avoid mismatch
    def compress(self, x, dpb, is_reset):
        """Compress one P-frame.

        Args:
            x: [1, 3, H, W], in range [-1, 1]
            dpb: decoded picture buffer
            is_reset: True for the first P-frame after an I-frame.
        """
        device = next(self.parameters()).device
        z_index = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        assert x.shape[0] == 1, "B=1 is required for compression."

        x = self.pix_emb(x.to(device))

        feature_prev = self.apply_feature_adaptor(dpb, is_reset)
        ctx, ctx_t = self.feature_extractor(feature_prev)

        y = self.encoder(x, ctx)

        hyper_inp = self.pad_for_y(y)
        z = self.hyper_encoder(hyper_inp)
        z_hat, z_hat_write = round_and_to_int8(z)

        params = self.res_prior_param_decoder(z_hat, ctx_t)
        y_q, scales, y_hat = self.compress_dual_prior(
            y, params, self.y_spatial_prior)

        feature, f_pix, f_semantic = self.get_recon_and_feature(
            y_hat, ctx, feature_prev
        )

        self.entropy_coder.reset()
        self.bit_estimator_z.encode_z(z_hat_write, z_index)
        self.gaussian_encoder.encode_y(y_q, scales, skip_thres=self.force_zero_thres)
        self.entropy_coder.flush()
        bit_stream = self.entropy_coder.get_encoded_stream()

        torch.cuda.synchronize(device=device)
        return {
            "dpb": {
                "ref_frame": None,
                "ref_feature": feature,
            },
            "f_pix": f_pix,
            "f_semantic": f_semantic,
            "bit_stream": bit_stream,
        }

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float32)        # must be float32 to avoid mismatch
    def decompress(self, bit_stream, dpb, height, width, is_reset):
        device = next(self.parameters()).device
        z_index = torch.zeros(1, dtype=torch.long, device=device)

        self.entropy_coder.reset()
        self.entropy_coder.set_stream(bit_stream)
        z_size = self.get_downsampled_shape(height, width, 64)
        self.bit_estimator_z.decode_z(z_size, z_index)

        feature_prev = self.apply_feature_adaptor(dpb, is_reset)
        ctx, ctx_t = self.feature_extractor(feature_prev)

        z_hat = self.bit_estimator_z.get_z(z_size, device, torch.float32)
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        infos = self.decompress_dual_prior_part1(params)
        y_hat = self.decompress_dual_prior_part2(params, self.y_spatial_prior, infos)

        feature, f_pix, f_semantic = self.get_recon_and_feature(
            y_hat, ctx, feature_prev)

        return {
            "dpb": {
                "ref_frame": None,
                "ref_feature": feature,
            },
            "f_pix": f_pix,
            "f_semantic": f_semantic,
        }
