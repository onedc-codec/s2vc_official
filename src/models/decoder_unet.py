from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import LoraConfig
from diffusers import UNet2DConditionModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils import USE_PEFT_BACKEND, deprecate, scale_lora_layers, unscale_lora_layers

from modules.dcvc import DepthConvBlock5


class reduce_resblock(nn.Module):
    """Channel adjustment block placed before SD UNet conv_in."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.short_cut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0) \
            if in_ch != out_ch else nn.Identity()
        self.blocks = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.blocks(x) + self.short_cut(x)


class MotionFusionBlock(nn.Module):
    """Fuse current feature with previous frame feature.

    Used in P-frame UNet only. ``motion_ch`` channels of zeros are concatenated
    to keep the architecture identical to the training graph (the original
    motion branch fed an estimated optical flow there).
    """
    def __init__(self, in_ch, cond_ch, motion_ch):
        super().__init__()
        self.motion_ch = motion_ch
        self.cond_reduce = DepthConvBlock5(cond_ch, cond_ch // 2)
        cond_ch = cond_ch // 2
        self.prev_warp = nn.Sequential(
            DepthConvBlock5(cond_ch + motion_ch, cond_ch + motion_ch),
            DepthConvBlock5(cond_ch + motion_ch, cond_ch),
        )
        self.fusion = nn.Sequential(
            DepthConvBlock5(in_ch + cond_ch, in_ch + cond_ch),
            DepthConvBlock5(in_ch + cond_ch, in_ch),
        )
        self.zero_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.zero_conv.weight.data.zero_()

    def forward(self, x, cond):
        B, C, H, W = x.shape
        dummy_motion = torch.zeros(B, self.motion_ch, H, W, device=x.device, dtype=x.dtype)
        cond = self.cond_reduce(cond)
        prev_feat = self.prev_warp(torch.cat([cond, dummy_motion], dim=1))
        fusion_feat = self.fusion(torch.cat([x, prev_feat], dim=1))
        return x + self.zero_conv(fusion_feat)


def forward_unet(
    self,
    sample: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    prev_feats: Optional[Tuple[torch.Tensor]] = None,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
) -> Union[UNet2DConditionOutput, Tuple]:
    """Hacked SD1.5 UNet forward.

    Differences from the upstream forward:
      * applies ``self.vae_reduction`` to ``sample`` first to emit a 4-channel
        epsilon estimate (the original UNet returns the same shape as input);
      * when the active adapter is ``"pframe"``, fuses ``prev_feats`` into the
        first three down blocks via the ``pframe_motion_fusion`` submodules
        added by ``prepare_unet_for_codec``;
      * also collects per-block features in ``out_prev_feats`` so the caller
        can feed them into the next frame's forward.
    Returns ``(sample, reduced_sample, out_prev_feats)``.
    """
    default_overall_up_factor = 2 ** self.num_upsamplers

    input_sample = sample
    reduced_sample = self.vae_reduction(input_sample)

    forward_upsample_size = False
    upsample_size = None

    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0:
            forward_upsample_size = True
            break

    if attention_mask is not None:
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    if encoder_attention_mask is not None:
        encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb = self.time_embedding(t_emb, timestep_cond)

    class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
    if class_emb is not None:
        if self.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    aug_emb = self.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )
    if self.config.addition_embed_type == "image_hint":
        aug_emb, hint = aug_emb
        sample = torch.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    encoder_hidden_states = self.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    # 2. pre-process
    sample = self.conv_in(sample)

    # decide whether this forward is an I-frame or a P-frame call
    is_fuse_motion = False
    active = getattr(self, "active_adapters", lambda: [])()
    if ("iframe" in active) or (active == "iframe"):
        assert prev_feats is None, "I-frame path must not receive prev features."
    elif ("pframe" in active) or (active == "pframe"):
        assert prev_feats is not None, "P-frame path must receive prev features."
        is_fuse_motion = True
    else:
        raise ValueError("Either iframe or pframe should be set to active adapters.")

    out_prev_feats = []

    # 2.5 GLIGEN position net
    if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        gligen_args = cross_attention_kwargs.pop("gligen")
        cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

    # 3. down
    if cross_attention_kwargs is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        lora_scale = cross_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
    is_adapter = down_intrablock_additional_residuals is not None
    if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
        deprecate(
            "T2I should not use down_block_additional_residuals",
            "1.3.0",
            "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated "
            "and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used "
            "for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
            standard_warn=False,
        )
        down_intrablock_additional_residuals = down_block_additional_residuals
        is_adapter = True

    down_block_res_samples = (sample,)
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            additional_residuals = {}
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                sample += down_intrablock_additional_residuals.pop(0)

        # motion propagation & fusion part.
        if hasattr(downsample_block, "pframe_motion_fusion"):
            out_prev_feats.append(sample)
            if is_fuse_motion:
                prev_cond_this = prev_feats.pop(0)
                sample = downsample_block.pframe_motion_fusion(sample, prev_cond_this)

        down_block_res_samples += res_samples

    if is_controlnet:
        new_down_block_res_samples = ()

        for down_block_res_sample, down_block_additional_residual in zip(
            down_block_res_samples, down_block_additional_residuals
        ):
            down_block_res_sample = down_block_res_sample + down_block_additional_residual
            new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = new_down_block_res_samples

    # 4. mid
    if self.mid_block is not None:
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = self.mid_block(sample, emb)

        if (
            is_adapter
            and len(down_intrablock_additional_residuals) > 0
            and sample.shape == down_intrablock_additional_residuals[0].shape
        ):
            sample += down_intrablock_additional_residuals.pop(0)

    if is_controlnet:
        sample = sample + mid_block_additional_residual

    # 5. up
    for i, upsample_block in enumerate(self.up_blocks):
        is_final_block = i == len(self.up_blocks) - 1

        res_samples = down_block_res_samples[-len(upsample_block.resnets):]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                upsample_size=upsample_size,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
            )

    # 6. post-process
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    return sample, reduced_sample, out_prev_feats


def unet_add_lora(unet: UNet2DConditionModel,
                  lora_rank: int = 64,
                  lora_alpha: float = 8.0,
                  lora_dropout: float = 0.0,
                  lora_name: str = "default"):
    lora_target_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "proj_in",
        "proj_out",
        "ff.net.0.proj",
        "ff.net.2",
        "conv1",
        "conv2",
        "conv_shortcut",
        "downsamplers.0.conv",
        "upsamplers.0.conv",
        "time_emb_proj",
    ]

    exclude_modules = []
    for name, _ in unet.named_modules():
        if "add_attentions" in name:
            exclude_modules.append(name)
        if "conv_in" in name:
            exclude_modules.append(name)

    lora_config = LoraConfig(
        r=lora_rank,
        target_modules=lora_target_modules,
        exclude_modules=exclude_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    unet.add_adapter(lora_config, adapter_name=lora_name)
    return unet


def prepare_unet_for_codec(in_ch: int, lora_config, motion_ch,
                           sd15_model_id: str = "runwayml/stable-diffusion-v1-5"):
    """Build the SD1.5 UNet variant used by the codec.

    Architecture changes:
        * input ``conv_in`` is replaced with a ``in_ch -> block_out_channels[0]``
          conv (input is no longer a 4-channel VAE latent but the codec
          feature ``f_pix``);
        * a ``vae_reduction`` head is added that maps the input back to the
          4-channel space — its output is treated as the predicted epsilon by
          ``forward_unet``;
        * two LoRA adapters ``iframe`` / ``pframe`` are installed;
        * a ``MotionFusionBlock`` is attached to the first three down blocks
          for P-frame motion fusion.

    The base SD1.5 UNet weights are pulled from ``sd15_model_id`` for
    initialisation only; the actual weights are overwritten by the user's
    checkpoint via ``load_state_dict(..., strict=False)``.
    """
    feedforward_model = UNet2DConditionModel.from_pretrained(
        sd15_model_id,
        subfolder="unet"
    ).float()

    # 1. add I-frame LoRA
    feedforward_model = unet_add_lora(
        feedforward_model,
        lora_rank=lora_config["lora_rank"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        lora_name="iframe"
    )

    # 2. add P-frame LoRA
    feedforward_model = unet_add_lora(
        feedforward_model,
        lora_rank=lora_config["lora_rank"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        lora_name="pframe"
    )

    # 3. replace conv_in
    vae_ch = feedforward_model.conv_in.in_channels
    ch_out = feedforward_model.conv_in.out_channels
    feedforward_model.conv_in = nn.Conv2d(in_ch, ch_out, kernel_size=3, stride=1, padding=1)

    # 4. add vae_reduction
    feedforward_model.add_module(
        "vae_reduction",
        reduce_resblock(in_ch, vae_ch),
    )

    # 5. hack the forward function
    feedforward_model.forward = forward_unet.__get__(feedforward_model)

    # 6. insert motion fusion block on the first three down blocks
    for i in range(3):
        feedforward_model.down_blocks[i].add_module(
            "pframe_motion_fusion",
            MotionFusionBlock(
                in_ch=feedforward_model.config.block_out_channels[i],
                cond_ch=feedforward_model.config.block_out_channels[i],
                motion_ch=motion_ch
            )
        )

    return feedforward_model
