# Copyright 2025 The Newton Developers
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

from typing import Optional

import mujoco
import warp as wp

from . import collision_driver
from . import constraint
from . import math
from . import passive
from . import smooth
from . import solver
from .support import xfrc_accumulate
from .types import MJ_MINVAL
from .types import BiasType
from .types import Data
from .types import DisableBit
from .types import DynType
from .types import GainType
from .types import JointType
from .types import Model
from .types import array2df
from .types import array3df
from .warp_util import event_scope
from .warp_util import kernel
from .warp_util import kernel_copy


def _advance(
  m: Model, d: Data, act_dot: wp.array, qacc: wp.array, qvel: Optional[wp.array] = None
):
  """Advance state and time given activation derivatives and acceleration."""


def update_actuator_state(m: Model, d: Data):
    """Support for stateful actuators."""
    for i in range(m.nu):
        if m.actuator_actlimited[i]:
            d.actuator_velocity[:, i] = wp.clip(d.actuator_velocity[:, i], m.actuator_actrange[i][0], m.actuator_actrange[i][1])
            # Additional logic for stateful actuators can be added here


@kernel
def sparse_actuator_dynamics(m: Model, d: Data):
    """Sparse version of actuator dynamics calculations."""
    # Implement sparse calculations for actuator dynamics
    pass


@kernel
def next_activation(
    m: Model,
    d: Data,
    act_dot_in: array2df,
):
    worldId, actid = wp.tid()

    # get the high/low range for each actuator state
    limited = m.actuator_actlimited[actid]
    range_low = wp.where(limited, m.actuator_actrange[actid][0], -wp.inf)
    range_high = wp.where(limited, m.actuator_actrange[actid][1], wp.inf)

    # get the actual actuation - skip if -1 (means stateless actuator)
    act_adr = m.actuator_actadr[actid]
    if act_adr == -1:
        return

    acts = d.act[worldId]
    acts_dot = act_dot_in[worldId]

    act = acts[act_adr]
    act_dot = acts_dot[act_adr]

    # check dynType
    dyn_type = m.actuator_dyntype[actid]
    dyn_prm = m.actuator_dynprm[actid][0]

    # advance the actuation
    if dyn_type == wp.static(DynType.FILTEREXACT.value):
        tau = wp.where(dyn_prm < MJ_MINVAL, MJ_MINVAL, dyn_prm)
        act = act + act_dot * tau * (1.0 - wp.exp(-m.opt.timestep / tau))
    else:
        act = act + act_dot * m.opt.timestep

    # apply limits
    wp.clamp(act, range_low, range_high)

    acts[act_adr] = act
