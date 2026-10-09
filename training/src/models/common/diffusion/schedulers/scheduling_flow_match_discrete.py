# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMatchDiscreteSchedulerOutput(BaseOutput):
    """Sample returned by one scheduler step."""

    prev_sample: torch.FloatTensor


class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """Euler scheduler for flow matching."""

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        reverse: bool = True,
        solver: str = "euler",
        n_tokens: Optional[int] = None,
    ):
        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)

        if not reverse:
            sigmas = sigmas.flip(0)

        self.sigmas = sigmas
        self.timesteps = (
            sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)

        self._step_index = None
        self._begin_index = None

        self.supported_solver = ["euler"]
        if solver not in self.supported_solver:
            raise ValueError(
                f"Solver {solver} not supported. Supported solvers: {self.supported_solver}"
            )

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
        n_tokens: int = None,
    ):
        """Set the inference timestep schedule."""
        self.num_inference_steps = num_inference_steps

        sigmas = torch.linspace(1, 0, num_inference_steps + 1)
        sigmas = self.sd3_time_shift(sigmas)

        if not self.config.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(
            dtype=torch.float32, device=device
        )

        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_model_input(
        self, sample: torch.Tensor, timestep: Optional[int] = None
    ) -> torch.Tensor:
        return sample

    def sd3_time_shift(self, t: torch.Tensor):
        return (self.config.shift * t) / (1 + (self.config.shift - 1) * t)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[FlowMatchDiscreteSchedulerOutput, Tuple]:
        """Advance the sample by one Euler step."""

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)

        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]

        if self.config.solver == "euler":
            prev_sample = sample + model_output.to(torch.float32) * dt
        else:
            raise ValueError(
                f"Solver {self.config.solver} not supported. Supported solvers: {self.supported_solver}"
            )

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
