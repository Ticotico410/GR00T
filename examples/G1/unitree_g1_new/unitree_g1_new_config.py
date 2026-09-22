# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modality config for Unitree G1 hangzhou joint-space datasets (e.g. arrange_flowers).

State (31 dims, matches meta/modality.json / observation.state):
  - robot_29dof: 29 joint angles (legs + waist + arms)
  - gripper:     2 gripper positions

Action (35 dims):
  - robot_29dof:  29 joint targets
  - gripper:      2 gripper targets
  - base_command: 4 (vx, vy, angle_z, height)

Video: head_stereo_left + wrist_left + wrist_right (no right stereo).

No SMPL / rot6d / euler root processing — train with default
``--root-process-mode original`` (see train_new.sh).
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

unitree_g1_new_config = {
    # Video: keys must match "video" entries in meta/modality.json
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "head_stereo_left",
            "wrist_left",
            "wrist_right",
        ],
    ),
    # State: robot_29dof (29) + gripper (2) = 31 dims
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "robot_29dof",
            "gripper",
        ],
    ),
    # Action: 50-step horizon @ 30 fps; 29 + 2 + 4 = 35 dims
    "action": ModalityConfig(
        delta_indices=list(range(0, 50)),
        modality_keys=[
            "robot_29dof",
            "gripper",
            "base_command",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: episode tasks from episodes.jsonl (LANG_KEYS "task")
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}

register_modality_config(unitree_g1_new_config, embodiment_tag=EmbodimentTag.UNITREE_G1_NEW)
