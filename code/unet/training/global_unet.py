"""Residual U-Net architecture for daily global 2-m temperature downscaling.
Defines periodic-longitude convolutions, optional pole-aware latitude padding,
residual blocks and bilinear upsampling. Correction mode adds the learned
residual to the bilinear input after converting it to target-normalized units.
The paper uses five inputs, base width 96, depth 6, GroupNorm and pole-aware
padding; these selected settings differ from some constructor defaults.
This module defines the network and does not run training or inference."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


POLE_AWARE_BOUNDARY_SCHEME = "conv3x3_and_bilinear_upsample_v1"


NormMode = Literal["none", "group"]
TargetMode = Literal["correction", "direct"]
LatitudePadding = Literal["replicate", "pole_aware"]


def _group_count(number_of_channels: int) -> int:


    for candidate in (8, 4, 2, 1):
        if number_of_channels % candidate == 0:
            return candidate

    return 1


def _make_normalization(
    number_of_channels: int,
    mode: NormMode,
) -> nn.Module:


    if mode == "none":
        return nn.Identity()

    if mode == "group":
        return nn.GroupNorm(
            num_groups=_group_count(number_of_channels),
            num_channels=number_of_channels,
        )

    raise ValueError(
        f"Unsupported normalization mode {mode!r}; "
        "expected 'none' or 'group'."
    )


class GlobalPeriodicConv2d(nn.Module):


    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        latitude_padding: LatitudePadding = "replicate",
    ) -> None:
        super().__init__()

        if input_channels < 1 or output_channels < 1:
            raise ValueError(
                "Convolution channel counts must be positive."
            )

        if latitude_padding not in ("replicate", "pole_aware"):
            raise ValueError(
                "Supported latitude-padding modes are 'replicate' and "
                f"'pole_aware', got {latitude_padding!r}."
            )

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.latitude_padding = latitude_padding

        self.convolution = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False,
        )

    @staticmethod
    def _pole_ghost_row(row: torch.Tensor) -> torch.Tensor:


        width = row.shape[-1]
        if width < 2:
            raise ValueError(
                "Pole-aware ghost rows need a longitude width of at "
                f"least 2, got {width}."
            )
        if width % 2 == 0:
            return torch.roll(row, shifts=width // 2, dims=-1)
        return (
            0.5 * torch.roll(row, shifts=width // 2, dims=-1)
            + 0.5 * torch.roll(row, shifts=width // 2 + 1, dims=-1)
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(
                "GlobalPeriodicConv2d expects a four-dimensional tensor, "
                f"got shape {tuple(tensor.shape)}."
            )

        if self.latitude_padding == "replicate":

            tensor = F.pad(
                tensor,
                pad=(1, 1, 0, 0),
                mode="circular",
            )


            tensor = F.pad(
                tensor,
                pad=(0, 0, 1, 1),
                mode=self.latitude_padding,
            )

            return self.convolution(tensor)


        south_ghost = self._pole_ghost_row(tensor[..., :1, :])
        north_ghost = self._pole_ghost_row(tensor[..., -1:, :])

        tensor = torch.cat(
            (south_ghost, tensor, north_ghost),
            dim=-2,
        )

        tensor = F.pad(
            tensor,
            pad=(1, 1, 0, 0),
            mode="circular",
        )

        return self.convolution(tensor)


class PeriodicBilinearUpsample2x(nn.Module):


    def __init__(self, latitude_padding: LatitudePadding = "replicate") -> None:
        super().__init__()
        if latitude_padding not in ("replicate", "pole_aware"):
            raise ValueError(
                "latitude_padding must be 'replicate' or "
                f"'pole_aware', got {latitude_padding!r}."
            )
        self.latitude_padding = latitude_padding

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(
                "PeriodicBilinearUpsample2x expects a four-dimensional "
                f"tensor, got shape {tuple(tensor.shape)}."
            )

        original_height = tensor.shape[-2]
        original_width = tensor.shape[-1]

        if self.latitude_padding == "replicate":
            longitude_padded = F.pad(
                tensor,
                pad=(1, 1, 0, 0),
                mode="circular",
            )

            upsampled = F.interpolate(
                longitude_padded,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )


            upsampled = upsampled[..., 2:-2]
        else:


            south_ghost = GlobalPeriodicConv2d._pole_ghost_row(
                tensor[..., :1, :]
            )
            north_ghost = GlobalPeriodicConv2d._pole_ghost_row(
                tensor[..., -1:, :]
            )
            latitude_padded = torch.cat(
                (south_ghost, tensor, north_ghost),
                dim=-2,
            )
            longitude_padded = F.pad(
                latitude_padded,
                pad=(1, 1, 0, 0),
                mode="circular",
            )

            upsampled = F.interpolate(
                longitude_padded,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )


            upsampled = upsampled[..., 2:-2, 2:-2]

        expected_shape = (
            original_height * 2,
            original_width * 2,
        )

        if upsampled.shape[-2:] != expected_shape:
            raise RuntimeError(
                "Periodic upsampling produced spatial shape "
                f"{tuple(upsampled.shape[-2:])}, expected "
                f"{expected_shape}."
            )

        return upsampled


class PreActivationResidualBlock(nn.Module):


    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        norm: NormMode,
        latitude_padding: LatitudePadding,
    ) -> None:
        super().__init__()

        if input_channels < 1 or output_channels < 1:
            raise ValueError(
                "Residual-block channel counts must be positive."
            )

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.norm_mode = norm
        self.latitude_padding = latitude_padding

        self.norm1 = _make_normalization(
            input_channels,
            norm,
        )
        self.activation1 = nn.SiLU()

        self.conv1 = GlobalPeriodicConv2d(
            input_channels,
            output_channels,
            latitude_padding=latitude_padding,
        )

        self.norm2 = _make_normalization(
            output_channels,
            norm,
        )
        self.activation2 = nn.SiLU()

        self.conv2 = GlobalPeriodicConv2d(
            output_channels,
            output_channels,
            latitude_padding=latitude_padding,
        )

        if input_channels == output_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )


        nn.init.zeros_(self.conv2.convolution.weight)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(tensor)

        residual = self.norm1(tensor)
        residual = self.activation1(residual)
        residual = self.conv1(residual)

        residual = self.norm2(residual)
        residual = self.activation2(residual)
        residual = self.conv2(residual)

        return shortcut + residual


class GlobalUNet(nn.Module):


    HEAD_INITIALIZATION_GAIN = 1.0e-3

    def __init__(
        self,
        *,
        in_channels: int = 3,
        base: int = 32,
        depth: int = 4,
        norm: NormMode = "none",
        target_mode: TargetMode = "correction",
        lat_padding: LatitudePadding = "replicate",
        input_t2m_mean: float,
        input_t2m_std: float,
        target_t2m_mean: float,
        target_t2m_std: float,
    ) -> None:
        super().__init__()

        if in_channels not in (3, 5):
            raise ValueError(
                "The current experiment supports exactly 3 or 5 input "
                f"channels, got {in_channels}."
            )

        if base < 1:
            raise ValueError(
                f"base must be positive, got {base}."
            )

        if depth < 1:
            raise ValueError(
                f"depth must be positive, got {depth}."
            )

        if norm not in ("none", "group"):
            raise ValueError(
                f"Unsupported norm mode {norm!r}."
            )

        if target_mode not in ("correction", "direct"):
            raise ValueError(
                f"Unsupported target mode {target_mode!r}."
            )

        if lat_padding not in ("replicate", "pole_aware"):
            raise ValueError(
                "Supported latitude-padding modes are 'replicate' and "
                f"'pole_aware', got {lat_padding!r}."
            )

        normalization_values = (
            input_t2m_mean,
            input_t2m_std,
            target_t2m_mean,
            target_t2m_std,
        )

        if not all(
            torch.isfinite(torch.tensor(value)).item()
            for value in normalization_values
        ):
            raise ValueError(
                "All T2M normalization values must be finite."
            )

        if input_t2m_std <= 0.0:
            raise ValueError(
                "input_t2m_std must be positive."
            )

        if target_t2m_std <= 0.0:
            raise ValueError(
                "target_t2m_std must be positive."
            )

        self.in_channels = in_channels
        self.base = base
        self.depth = depth
        self.norm_mode: NormMode = norm
        self.target_mode: TargetMode = target_mode
        self.lat_padding: LatitudePadding = lat_padding


        self.register_buffer(
            "input_t2m_mean",
            torch.tensor(
                float(input_t2m_mean),
                dtype=torch.float64,
            ),
        )
        self.register_buffer(
            "input_t2m_std",
            torch.tensor(
                float(input_t2m_std),
                dtype=torch.float64,
            ),
        )
        self.register_buffer(
            "target_t2m_mean",
            torch.tensor(
                float(target_t2m_mean),
                dtype=torch.float64,
            ),
        )
        self.register_buffer(
            "target_t2m_std",
            torch.tensor(
                float(target_t2m_std),
                dtype=torch.float64,
            ),
        )

        baseline_scale = (
            float(input_t2m_std)
            / float(target_t2m_std)
        )
        baseline_offset = (
            float(input_t2m_mean)
            - float(target_t2m_mean)
        ) / float(target_t2m_std)


        self.register_buffer(
            "baseline_scale",
            torch.tensor(
                baseline_scale,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "baseline_offset",
            torch.tensor(
                baseline_offset,
                dtype=torch.float32,
            ),
        )


        self.widths = tuple(
            min(base * (2**level), base * 8)
            for level in range(depth + 1)
        )

        self.encoder_blocks = nn.ModuleList()

        current_channels = in_channels


        for output_channels in self.widths[:-1]:
            self.encoder_blocks.append(
                PreActivationResidualBlock(
                    current_channels,
                    output_channels,
                    norm=norm,
                    latitude_padding=lat_padding,
                )
            )
            current_channels = output_channels

        self.pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
        )

        self.bottleneck = PreActivationResidualBlock(
            current_channels,
            self.widths[-1],
            norm=norm,
            latitude_padding=lat_padding,
        )

        current_channels = self.widths[-1]

        self.upsample = PeriodicBilinearUpsample2x(latitude_padding=lat_padding)
        self.decoder_projections = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for skip_channels in reversed(self.widths[:-1]):
            self.decoder_projections.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=skip_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            )

            self.decoder_blocks.append(
                PreActivationResidualBlock(
                    input_channels=2 * skip_channels,
                    output_channels=skip_channels,
                    norm=norm,
                    latitude_padding=lat_padding,
                )
            )

            current_channels = skip_channels

        self.head = nn.Conv2d(
            in_channels=current_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )


        nn.init.xavier_uniform_(
            self.head.weight,
            gain=self.HEAD_INITIALIZATION_GAIN,
        )
        nn.init.zeros_(self.head.bias)

    def _validate_inputs(
        self,
        inputs: torch.Tensor,
    ) -> None:
        if inputs.ndim != 4:
            raise ValueError(
                "Expected input shape "
                "(batch, channels, latitude, longitude), got "
                f"{tuple(inputs.shape)}."
            )

        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {inputs.shape[1]}."
            )

        factor = 2**self.depth
        height, width = inputs.shape[-2:]

        if height % factor != 0 or width % factor != 0:
            raise ValueError(
                f"Spatial dimensions {(height, width)} must both be "
                f"divisible by 2**depth={factor}."
            )

        if not inputs.is_floating_point():
            raise TypeError(
                f"Model inputs must be floating point, got {inputs.dtype}."
            )

    def baseline_in_target_units(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:


        self._validate_inputs(inputs)

        input_t2m = inputs[:, 0:1]

        return (
            input_t2m * self.baseline_scale
            + self.baseline_offset
        )

    def network_output(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:


        self._validate_inputs(inputs)

        tensor = inputs
        skip_connections: list[torch.Tensor] = []

        for encoder_block in self.encoder_blocks:
            tensor = encoder_block(tensor)
            skip_connections.append(tensor)
            tensor = self.pool(tensor)

        tensor = self.bottleneck(tensor)

        for projection, decoder_block, skip in zip(
            self.decoder_projections,
            self.decoder_blocks,
            reversed(skip_connections),
        ):
            tensor = self.upsample(tensor)
            tensor = projection(tensor)

            if tensor.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError(
                    "Decoder and encoder-skip spatial shapes differ: "
                    f"{tuple(tensor.shape[-2:])} versus "
                    f"{tuple(skip.shape[-2:])}."
                )

            tensor = torch.cat(
                (tensor, skip),
                dim=1,
            )

            tensor = decoder_block(tensor)

        return self.head(tensor)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        learned_field = self.network_output(inputs)

        if self.target_mode == "direct":
            return learned_field

        baseline = self.baseline_in_target_units(inputs)

        return baseline + learned_field

    def architecture_dict(self) -> dict[str, object]:


        if self.lat_padding == "pole_aware":
            return {
                **self._architecture_dict_base(),
                "pole_aware_boundary_scheme": POLE_AWARE_BOUNDARY_SCHEME,
            }
        return self._architecture_dict_base()

    def _architecture_dict_base(self) -> dict[str, object]:
        return {
            "name": "GlobalUNet",
            "in_channels": self.in_channels,
            "output_channels": 1,
            "base": self.base,
            "depth": self.depth,
            "widths": list(self.widths),
            "norm": self.norm_mode,
            "target_mode": self.target_mode,
            "latitude_padding": self.lat_padding,
            "longitude_padding": "circular",
            "pooling": "average_2x2",
            "upsampling": (
                "periodic_bilinear_2x_align_corners_false"
            ),
            "block": (
                "preactivation_two_conv_residual_"
                "zero_initialized_second_conv_"
                "no_post_add_activation"
            ),
            "head_initialization_gain": (
                self.HEAD_INITIALIZATION_GAIN
            ),
            "input_t2m_mean": float(
                self.input_t2m_mean.item()
            ),
            "input_t2m_std": float(
                self.input_t2m_std.item()
            ),
            "target_t2m_mean": float(
                self.target_t2m_mean.item()
            ),
            "target_t2m_std": float(
                self.target_t2m_std.item()
            ),
            "baseline_scale": float(
                self.baseline_scale.item()
            ),
            "baseline_offset": float(
                self.baseline_offset.item()
            ),
        }
