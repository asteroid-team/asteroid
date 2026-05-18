from typing import Optional

import torch
from torch import nn

from ..masknn.hailo_convolutional import HailoTDConvNet2D


class HailoEncoder2D(nn.Module):
    def __init__(self, n_filters: int = 256, kernel_size: int = 16, stride: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(
            1,
            n_filters,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, kernel_size // 2),
            bias=False,
        )

    def forward(self, wav_4d: torch.Tensor) -> torch.Tensor:
        return self.conv(wav_4d)


class HailoDecoderConv1x1Head(nn.Module):
    def __init__(self, in_chan: int, out_chan: int):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HailoConvTasNet(nn.Module):
    """Parallel Hailo-safe ConvTasNet variant for module-by-module HAR migration."""

    def __init__(
        self,
        n_src: int = 2,
        n_filters: int = 256,
        kernel_size: int = 16,
        stride: int = 8,
        bn_chan: int = 128,
        hid_chan: int = 256,
        skip_chan: int = 128,
        n_blocks: int = 1,
        n_repeats: int = 1,
        conv_kernel_size: int = 3,
        mask_act: str = "sigmoid",
        norm_mode: str = "affine",
        mask_mul_mode: str = "normal",
        force_n_src_1: bool = False,
        skip_topology_mode: str = "project",
        decoder_mode: str = "conv1x1_head",
        truncate_k_blocks: int = 1,
    ):
        super().__init__()
        self.model_n_src = n_src
        self.export_n_src = 1 if force_n_src_1 else n_src
        self.mask_mul_mode = mask_mul_mode
        self.skip_topology_mode = skip_topology_mode
        self.decoder_mode = decoder_mode

        self.encoder = HailoEncoder2D(n_filters=n_filters, kernel_size=kernel_size, stride=stride)
        self.masker = HailoTDConvNet2D(
            in_chan=n_filters,
            n_src=self.export_n_src,
            out_chan=n_filters,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            skip_chan=skip_chan,
            conv_kernel_size=conv_kernel_size,
            mask_act=mask_act,
            norm_mode=norm_mode,
            disable_skip=False,
            truncate_k_blocks=truncate_k_blocks,
        )

        in_total = n_filters * self.export_n_src
        out_total = self.export_n_src

        self.source_projector: Optional[nn.Conv2d] = None
        if self.export_n_src > 1:
            proj = nn.Conv2d(n_filters, in_total, kernel_size=1, bias=False)
            with torch.no_grad():
                proj.weight.zero_()
                for src_idx in range(self.export_n_src):
                    start = src_idx * n_filters
                    for ch in range(n_filters):
                        proj.weight[start + ch, ch, 0, 0] = 1.0
            self.source_projector = proj

        if decoder_mode == "conv1x1_head":
            self.decoder_pre = nn.Identity()
            self.decoder = HailoDecoderConv1x1Head(in_total, out_total)
        elif decoder_mode in {"reduced_deconv_64", "reduced_deconv_128"}:
            reduced_ch = 64 if decoder_mode.endswith("64") else 128
            self.decoder_pre = nn.Conv2d(in_total, reduced_ch, kernel_size=1, bias=False)
            self.decoder = nn.ConvTranspose2d(
                reduced_ch,
                out_total,
                kernel_size=(1, kernel_size),
                stride=(1, stride),
                padding=(0, kernel_size // 2),
                groups=1,
                bias=True,
            )
        else:
            raise ValueError(f"Unsupported Hailo decoder mode: {decoder_mode}")

    @staticmethod
    def _to_4d(wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 2:
            return wav.unsqueeze(1).unsqueeze(2)
        if wav.dim() == 3:
            return wav.unsqueeze(2)
        if wav.dim() == 4:
            return wav
        raise ValueError(f"Unsupported input shape: {tuple(wav.shape)}")

    def _expand_tf(self, tf_rep: torch.Tensor) -> torch.Tensor:
        if self.export_n_src == 1:
            return tf_rep
        if self.skip_topology_mode == "project" and self.source_projector is not None:
            return self.source_projector(tf_rep)
        return tf_rep.repeat(1, self.export_n_src, 1, 1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav_4d = self._to_4d(wav)
        tf_rep = self.encoder(wav_4d)

        est_masks = self.masker(tf_rep)
        tf_rep_exp = self._expand_tf(tf_rep)

        if self.mask_mul_mode == "bypass":
            masked = tf_rep_exp
        else:
            masked = est_masks * tf_rep_exp

        decoded = self.decoder(self.decoder_pre(masked))
        # [N, n_src, 1, T] -> [N, n_src, T]
        return decoded.squeeze(2)
