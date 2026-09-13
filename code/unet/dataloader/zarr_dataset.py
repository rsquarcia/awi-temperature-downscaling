"""PyTorch dataset for reading complete daily AWI temperature fields from Zarr.
Loads normalized temperature, incoming solar radiation and orography, with
optional continuous land and lake fractions to form the five-channel input.
Returns target-normalized high-resolution temperature and manages Zarr handles
and static-field caches separately for each data-loader process.
Requires preprocessed stores and their expected shapes and metadata."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


DYNAMIC_INPUT_SHAPE = (3, 1280, 2624)
STATIC_INPUT_SHAPE = (2, 1280, 2624)
TARGET_SHAPE = (1, 1280, 2624)

EXPECTED_STATIC_CHANNEL_NAMES = [
    "lsm",
    "cl",
]


class AWIDownscalingZarrDataset(Dataset):


    def __init__(
        self,
        zarr_path: str | Path,
        return_metadata: bool = False,
        include_static_inputs: bool = False,
    ) -> None:
        self.zarr_path = str(Path(zarr_path))
        self.return_metadata = return_metadata
        self.include_static_inputs = include_static_inputs


        self._root: Any = None
        self._root_pid: int | None = None
        self._static_tensor: torch.Tensor | None = None
        self._static_pid: int | None = None


        root = zarr.open_group(
            self.zarr_path,
            mode="r",
        )

        required_arrays = {
            "inputs",
            "targets",
            "dates",
        }
        present_arrays = set(root.array_keys())

        if not required_arrays.issubset(
            present_arrays
        ):
            raise KeyError(
                f"Missing arrays. Found {sorted(present_arrays)}, "
                f"required at least {sorted(required_arrays)}"
            )

        if root.attrs.get("complete") is not True:
            raise ValueError(
                "Zarr store is not marked complete: "
                f"{self.zarr_path}"
            )

        inputs = root["inputs"]
        targets = root["targets"]
        dates = root["dates"]

        if inputs.ndim != 4:
            raise ValueError(
                "Inputs must have four dimensions, got "
                f"{inputs.shape}"
            )

        if targets.ndim != 4:
            raise ValueError(
                "Targets must have four dimensions, got "
                f"{targets.shape}"
            )

        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(
                "Inputs and targets contain different "
                "numbers of days"
            )

        if dates.shape != (inputs.shape[0],):
            raise ValueError(
                f"Dates shape {dates.shape} does not match "
                f"{inputs.shape[0]} samples"
            )

        if inputs.shape[1:] != DYNAMIC_INPUT_SHAPE:
            raise ValueError(
                "Unexpected dynamic input sample shape: "
                f"{inputs.shape[1:]}"
            )

        if targets.shape[1:] != TARGET_SHAPE:
            raise ValueError(
                "Unexpected target sample shape: "
                f"{targets.shape[1:]}"
            )

        if inputs.dtype != np.dtype("float32"):
            raise TypeError(
                f"Expected float32 inputs, got {inputs.dtype}"
            )

        if targets.dtype != np.dtype("float32"):
            raise TypeError(
                f"Expected float32 targets, got {targets.dtype}"
            )

        self._length = inputs.shape[0]
        self.dynamic_input_shape = inputs.shape[1:]
        self.target_shape = targets.shape[1:]

        dynamic_channel_names = list(
            root.attrs.get(
                "input_channel_names",
                [],
            )
        )

        if dynamic_channel_names != [
            "t2m_inp",
            "tisr",
            "orography",
        ]:
            raise ValueError(
                "Unexpected dynamic input channel names: "
                f"{dynamic_channel_names}"
            )

        self.dynamic_input_channel_names = (
            dynamic_channel_names
        )

        self.target_channel_names = list(
            root.attrs.get(
                "target_channel_names",
                [],
            )
        )

        if self.target_channel_names != [
            "t2m_tgt"
        ]:
            raise ValueError(
                "Unexpected target channel names: "
                f"{self.target_channel_names}"
            )

        self.static_input_shape: (
            tuple[int, int, int] | None
        ) = None
        self.static_input_channel_names: list[str] = []

        if self.include_static_inputs:
            self._validate_static_structure(
                root,
                present_arrays,
            )

            self.static_input_shape = (
                STATIC_INPUT_SHAPE
            )
            self.static_input_channel_names = (
                EXPECTED_STATIC_CHANNEL_NAMES.copy()
            )

            self.input_shape = (
                5,
                1280,
                2624,
            )
            self.input_channel_names = (
                self.dynamic_input_channel_names
                + self.static_input_channel_names
            )
        else:
            self.input_shape = (
                DYNAMIC_INPUT_SHAPE
            )
            self.input_channel_names = (
                self.dynamic_input_channel_names.copy()
            )

    @staticmethod
    def _validate_static_structure(
        root: Any,
        present_arrays: set[str],
    ) -> None:


        if (
            root.attrs.get(
                "static_inputs_complete"
            )
            is not True
        ):
            raise ValueError(
                "Five-channel inputs were requested, but "
                "static_inputs_complete is not true"
            )

        if "static_inputs" not in present_arrays:
            raise KeyError(
                "Five-channel inputs were requested, but "
                "the static_inputs array is missing"
            )

        static_inputs = root["static_inputs"]

        if static_inputs.shape != STATIC_INPUT_SHAPE:
            raise ValueError(
                "Unexpected static input shape: "
                f"{static_inputs.shape}"
            )

        if static_inputs.dtype != np.dtype(
            "float32"
        ):
            raise TypeError(
                "Expected float32 static inputs, got "
                f"{static_inputs.dtype}"
            )

        root_names = list(
            root.attrs.get(
                "static_input_channel_names",
                [],
            )
        )

        array_names = list(
            static_inputs.attrs.get(
                "channel_names",
                [],
            )
        )

        if (
            root_names
            != EXPECTED_STATIC_CHANNEL_NAMES
        ):
            raise ValueError(
                "Unexpected root static channel names: "
                f"{root_names}"
            )

        if (
            array_names
            != EXPECTED_STATIC_CHANNEL_NAMES
        ):
            raise ValueError(
                "Unexpected static-array channel names: "
                f"{array_names}"
            )

        expected_five_channel_names = [
            "t2m_inp",
            "tisr",
            "orography",
            "lsm",
            "cl",
        ]

        observed_five_channel_names = list(
            root.attrs.get(
                "model_input_channel_names_5ch",
                [],
            )
        )

        if (
            observed_five_channel_names
            != expected_five_channel_names
        ):
            raise ValueError(
                "Unexpected five-channel model input names: "
                f"{observed_five_channel_names}"
            )

    def _get_root(self) -> Any:


        current_pid = os.getpid()

        if (
            self._root is None
            or self._root_pid != current_pid
        ):
            self._root = zarr.open_group(
                self.zarr_path,
                mode="r",
            )
            self._root_pid = current_pid


            self._static_tensor = None
            self._static_pid = None

        return self._root

    def _get_static_tensor(
        self,
    ) -> torch.Tensor:


        if not self.include_static_inputs:
            raise RuntimeError(
                "Static inputs were not enabled for this dataset"
            )

        root = self._get_root()
        current_pid = os.getpid()


        if (
            root.attrs.get(
                "static_inputs_complete"
            )
            is not True
        ):
            raise RuntimeError(
                "static_inputs_complete is not true"
            )

        if "static_inputs" not in set(
            root.array_keys()
        ):
            raise RuntimeError(
                "static_inputs disappeared from the store"
            )

        if (
            self._static_tensor is None
            or self._static_pid != current_pid
        ):
            static_numpy = np.asarray(
                root["static_inputs"][:],
                dtype=np.float32,
            )

            if (
                static_numpy.shape
                != STATIC_INPUT_SHAPE
            ):
                raise ValueError(
                    "Unexpected static input shape at "
                    f"read time: {static_numpy.shape}"
                )

            if not np.isfinite(
                static_numpy
            ).all():
                raise ValueError(
                    "Static inputs contain NaN or Inf"
                )

            if (
                static_numpy.min() < 0.0
                or static_numpy.max() > 1.0
            ):
                raise ValueError(
                    "Static inputs contain values "
                    "outside [0,1]"
                )

            static_numpy = np.ascontiguousarray(
                static_numpy
            )

            self._static_tensor = (
                torch.from_numpy(static_numpy)
            )
            self._static_tensor.requires_grad_(
                False
            )
            self._static_pid = current_pid

        return self._static_tensor

    def __len__(self) -> int:

        return self._length

    def __getitem__(
        self,
        index: int,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | dict[str, Any]
    ):

        if not isinstance(
            index,
            (int, np.integer),
        ):
            raise TypeError(
                "Index must be an integer, got "
                f"{type(index)}"
            )


        if index < 0:
            index += self._length

        if index < 0 or index >= self._length:
            raise IndexError(
                f"Index {index} outside dataset of "
                f"length {self._length}"
            )

        root = self._get_root()

        dynamic_numpy = np.asarray(
            root["inputs"][index],
            dtype=np.float32,
        )

        target_numpy = np.asarray(
            root["targets"][index],
            dtype=np.float32,
        )

        dynamic_tensor = torch.from_numpy(
            dynamic_numpy
        )
        target_tensor = torch.from_numpy(
            target_numpy
        )

        if self.include_static_inputs:
            input_tensor = torch.cat(
                (
                    dynamic_tensor,
                    self._get_static_tensor(),
                ),
                dim=0,
            )
        else:
            input_tensor = dynamic_tensor

        if self.return_metadata:
            stored_date = root["dates"][index]

            if isinstance(stored_date, bytes):
                stored_date = stored_date.decode(
                    "ascii"
                )
            else:
                stored_date = str(stored_date)

            return {
                "inputs": input_tensor,
                "targets": target_tensor,
                "date": stored_date,
                "index": int(index),
            }

        return input_tensor, target_tensor

    def __getstate__(self) -> dict[str, Any]:


        state = self.__dict__.copy()

        state["_root"] = None
        state["_root_pid"] = None
        state["_static_tensor"] = None
        state["_static_pid"] = None

        return state
