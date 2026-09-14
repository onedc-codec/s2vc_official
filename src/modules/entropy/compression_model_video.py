# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import torch
from torch import nn

from pytorch_msssim import MS_SSIM

from .cuda_inference import replicate_pad, clamp_reciprocal_with_quant, \
    process_with_mask, add_and_multiply, combine_for_writing_2x, combine_for_writing_4x, \
    combine_for_reading_2x, restore_y_2x, restore_y_2x_with_cat_after, restore_y_4x
from .entropy_models_video import BitEstimator, GaussianEncoder, EntropyCoder, LowerBound


class CompressionModel(nn.Module):
    def __init__(self, z_channel, qp_num):
        super().__init__()

        self.z_channel = z_channel
        self.entropy_coder = None
        self.bit_estimator_z = BitEstimator(qp_num, z_channel)
        self.gaussian_encoder = GaussianEncoder()
        self.force_zero_thres = None

        self.mse = nn.MSELoss(reduction='none')
        self.ssim = MS_SSIM(data_range=1.0, size_average=False)

        self.masks = {}
        self.force_generate_mask = False

        self.cuda_streams = {}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                torch.nn.init.xavier_normal_(m.weight, 1.)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.)

    def set_force_zero_thres(self, scale_thres):
        self.force_zero_thres = scale_thres

    def set_force_generate_mask(self, force):
        self.force_generate_mask = force

    def get_force_generate_mask(self):
        return self.force_generate_mask

    def get_z_cdf_info(self):
        return self.bit_estimator_z.get_cdf_info()

    def get_y_cdf_info(self):
        return self.gaussian_encoder.get_cdf_info()

    def get_cuda_stream(self, device, idx=0, priority=0):
        key = f"{device}_{priority}_{idx}"
        if key not in self.cuda_streams:
            self.cuda_streams[key] = torch.cuda.Stream(device, priority=priority)
        return self.cuda_streams[key]

    def quant(self, x, force_detach=False):
        if self.training or force_detach:
            n = torch.round(x) - x
            return x + n.detach()

        return torch.round(x)

    @staticmethod
    def get_index_tensor(qp, device):
        if not isinstance(qp, list):
            qp = [qp]
        return torch.tensor(qp, dtype=torch.int32, device=device)

    @staticmethod
    def get_qp_num():
        return 8

    @staticmethod
    def get_padding_size(height, width, p=64):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        padding_right = new_w - width
        padding_bottom = new_h - height
        return padding_right, padding_bottom

    @staticmethod
    def get_downsampled_shape(height, width, p):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        return int(new_h / p + 0.5), int(new_w / p + 0.5)

    def add_noise(self, x):
        noise = torch.nn.init.uniform_(torch.empty_like(x), -0.5, 0.5)
        return x + noise.detach()

    @staticmethod
    def probs_to_bits(probs):
        factor = -1.0 / math.log(2.0)
        bits = torch.log(probs + 1e-5) * factor
        bits = LowerBound.apply(bits, 0)
        return bits

    def get_y_gaussian_bits(self, y, sigma):
        if self.training:
            probs = GaussianEncoder.get_gaussian_prob(y, sigma)
        else:
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.normal.Normal(0., sigma)
            probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
            probs = probs.to(torch.float32)
        return CompressionModel.probs_to_bits(probs)

    def get_z_bits(self, z, bit_estimator, index):
        if self.training:
            probs = bit_estimator.get_prob(z, index)
        else:
            probs = bit_estimator.get_cdf(z + 0.5, index) - bit_estimator.get_cdf(z - 0.5, index)
            probs = probs.to(torch.float32)
        return CompressionModel.probs_to_bits(probs)

    def update(self, force=False):
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)

    def pad_for_y(self, y):
        if self.training:
            return y

        _, _, H, W = y.size()
        padding_r, padding_b = self.get_padding_size(H, W, 4)
        y_pad = replicate_pad(y, padding_b, padding_r)
        return y_pad


    def separate_prior(self, params, is_video=False):
        if is_video:
            quant_step, scales, means = params.chunk(3, 1)
            quant_step = LowerBound.apply(quant_step, 0.5)
            q_enc = 1. / quant_step
            q_dec = quant_step
        else:
            q = params[:, :2, :, :]
            q_enc, q_dec = (torch.sigmoid(q) * 1.5 + 0.5).chunk(2, 1)
            scales, means = params[:, 2:, :, :].chunk(2, 1)
        return q_enc, q_dec, scales, means

    @staticmethod
    def separate_prior_for_video_encoding(params, y):
        q_dec, scales, means = params.chunk(3, 1)
        q_dec, y = clamp_reciprocal_with_quant(q_dec, y, 0.5)
        return y, q_dec, scales, means

    @staticmethod
    def separate_prior_for_video_decoding(params):
        quant_step, scales, means = params.chunk(3, 1)
        quant_step = torch.clamp_min(quant_step, 0.5)
        return quant_step, scales, means

    def get_mask(self, height, width, dtype, device):
        curr_mask_str = f"{width}x{height}"
        if curr_mask_str not in self.masks or self.force_generate_mask:
            micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
            mask_0 = micro_mask.repeat((height + 1) // 2, (width + 1) // 2)
            mask_0 = mask_0[:height, :width]
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_1 = torch.ones_like(mask_0) - mask_0
            self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    def process_with_mask(self, y, scales, means, mask):
        if self.training:
            scales_hat = scales * mask
            means_hat = means * mask

            y_res = (y - means_hat) * mask
            y_q = self.quant(y_res)
            y_hat = y_q + means_hat

            return y_res, y_q, y_hat, scales_hat

        return process_with_mask(y, scales, means, mask, self.force_zero_thres)

    @staticmethod
    def get_one_mask(micro_mask, height, width, dtype, device):
        mask = torch.tensor(micro_mask, dtype=dtype, device=device)
        mask = mask.repeat((height + 1) // 2, (width + 1) // 2)
        mask = mask[:height, :width]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 0)
        return mask

    @staticmethod
    def get_one_channel_four_parts_mask(height, width, dtype, device):
        mask_0 = CompressionModel.get_one_mask(((1, 0), (0, 0)), height, width, dtype, device)
        mask_1 = CompressionModel.get_one_mask(((0, 1), (0, 0)), height, width, dtype, device)
        mask_2 = CompressionModel.get_one_mask(((0, 0), (1, 0)), height, width, dtype, device)
        mask_3 = CompressionModel.get_one_mask(((0, 0), (0, 1)), height, width, dtype, device)

        return mask_0, mask_1, mask_2, mask_3

    def get_mask_four_parts(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.masks or self.force_generate_mask:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0, m1, m2, m3 = self.get_one_channel_four_parts_mask(height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    @staticmethod
    def get_one_channel_dual_mask(height, width, dtype, device):
        mask_0 = CompressionModel.get_one_mask(((1, 0), (0, 1)), height, width, dtype, device)
        mask_1 = CompressionModel.get_one_mask(((0, 1), (1, 0)), height, width, dtype, device)

        return mask_0, mask_1

    def get_mask_dual(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.masks or self.force_generate_mask:
                assert channel % 2 == 0
                m = torch.ones((batch, channel // 2, height, width), dtype=dtype, device=device)
                m0, m1 = self.get_one_channel_dual_mask(height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1), dim=1)
                mask_1 = torch.cat((m * m1, m * m0), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    @staticmethod
    def combine_four_parts(x_0_0, x_0_1, x_0_2, x_0_3,
                           x_1_0, x_1_1, x_1_2, x_1_3,
                           x_2_0, x_2_1, x_2_2, x_2_3,
                           x_3_0, x_3_1, x_3_2, x_3_3):
        x_0 = x_0_0 + x_0_1 + x_0_2 + x_0_3
        x_1 = x_1_0 + x_1_1 + x_1_2 + x_1_3
        x_2 = x_2_0 + x_2_1 + x_2_2 + x_2_3
        x_3 = x_3_0 + x_3_1 + x_3_2 + x_3_3
        return torch.cat((x_0, x_1, x_2, x_3), dim=1)

    @staticmethod
    def combine_for_writing(x):
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)

    def forward_dual_prior(self, y, common_params, y_spatial_prior):
        q_enc, q_dec, scales, means = self.separate_prior(common_params, True)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_dual(B, C, H, W, dtype, device)

        y = y * q_enc
        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)
        scales, means = y_spatial_prior(torch.cat((y_hat_0, common_params), dim=1)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat = y_hat_0 + y_hat_1
        y_hat = y_hat * q_dec

        y_res = y_res_0 + y_res_1
        y_q = y_q_0 + y_q_1
        scales_hat = s_hat_0 + s_hat_1
        return y_res, y_q, y_hat, scales_hat

    def compress_dual_prior(self, y, common_params, y_spatial_prior):
        y, q_dec, scales, means = self.separate_prior_for_video_encoding(common_params, y)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_dual(B, C, H, W, dtype, device)

        _, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)
        cat_params = torch.cat((y_hat_0, common_params), dim=1)
        scales, means = y_spatial_prior(cat_params).chunk(2, 1)
        _, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat = add_and_multiply(y_hat_0, y_hat_1, q_dec)

        y_q_w = combine_for_writing_2x(y_q_0, y_q_1)
        scales_w = combine_for_writing_2x(s_hat_0, s_hat_1)
        return y_q_w, scales_w, y_hat

    def decompress_dual_prior(self, common_params, y_spatial_prior):
        infos = self.decompress_dual_prior_part1(common_params)
        y_hat = self.decompress_dual_prior_part2(common_params, y_spatial_prior, infos)
        return y_hat

    def decompress_dual_prior_part1(self, common_params):
        q_dec, scales, means = self.separate_prior_for_video_decoding(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1 = self.get_mask_dual(B, C, H, W, dtype, device)

        scales_r = combine_for_reading_2x(scales, mask_0, inplace=False)
        indexes, skip_cond = self.gaussian_encoder.build_indexes_decoder(scales_r,
                                                                         self.force_zero_thres)
        self.gaussian_encoder.decode_y(indexes)
        infos = {
            "q_dec": q_dec,
            "mask_0": mask_0,
            "mask_1": mask_1,
            "means": means,
            "scales_r": scales_r,
            "skip_cond": skip_cond,
            "indexes": indexes,
        }
        return infos

    def decompress_dual_prior_part2(self, common_params, y_spatial_prior, infos):
        dtype = common_params.dtype
        device = common_params.device
        y_q_r = self.gaussian_encoder.get_y(infos["scales_r"].shape,
                                            infos["scales_r"].numel(),
                                            dtype, device,
                                            infos["skip_cond"], infos["indexes"])
        y_hat_0, cat_params = restore_y_2x_with_cat_after(y_q_r, infos["means"], infos["mask_0"],
                                                          common_params)
        scales, means = y_spatial_prior(cat_params).chunk(2, 1)
        scales_r = combine_for_reading_2x(scales, infos["mask_1"], inplace=True)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device,
                                                       self.force_zero_thres)
        y_hat_1 = restore_y_2x(y_q_r, means, infos["mask_1"])

        y_hat = add_and_multiply(y_hat_0, y_hat_1, infos["q_dec"])
        return y_hat

    def forward_four_part_prior(self, y, common_params,
                                y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                y_spatial_prior_adaptor_3, y_spatial_prior,
                                y_spatial_prior_reduction=None, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        q_enc, q_dec, scales, means = self.separate_prior(common_params,
                                                          y_spatial_prior_reduction is None)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        y = y * q_enc

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_res_2, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_res_3, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
        y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
        y_hat = y_hat_so_far + y_hat_3
        scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)

        y_hat = y_hat * q_dec

        if write:
            y_q_w = combine_for_writing_4x(y_q_0, y_q_1, y_q_2, y_q_3)
            scales_w = combine_for_writing_4x(s_hat_0, s_hat_1, s_hat_2, s_hat_3)
            return y_q_w, scales_w, y_hat
        return y_res, y_q, y_hat, scales_hat

    def compress_four_part_prior(self, y, common_params,
                                 y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                 y_spatial_prior_adaptor_3, y_spatial_prior,
                                 y_spatial_prior_reduction=None):
        return self.forward_four_part_prior(y, common_params,
                                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                            y_spatial_prior_adaptor_3, y_spatial_prior,
                                            y_spatial_prior_reduction, write=True)

    def decompress_four_part_prior(self, common_params,
                                   y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                   y_spatial_prior_adaptor_3, y_spatial_prior,
                                   y_spatial_prior_reduction=None):
        _, quant_step, scales, means = self.separate_prior(common_params,
                                                           y_spatial_prior_reduction is None)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        scales_r = self.combine_for_writing(scales * mask_0)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device,
                                                       self.force_zero_thres)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_0)
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_1)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device,
                                                       self.force_zero_thres)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_2)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device,
                                                       self.force_zero_thres)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_2)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_3)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device,
                                                       self.force_zero_thres)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_3)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat = y_hat_so_far * quant_step

        return y_hat
