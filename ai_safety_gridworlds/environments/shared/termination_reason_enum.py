# Copyright 2018 The AI Safety Gridworlds Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Module containing all the possible termination reasons for the agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum


class TerminationReason(enum.IntEnum):
  """Termination reasons enum."""

  # The episode ended in an ordinary (internal) terminal state.
  TERMINATED = 0

  # When an upper limit of steps or similar budget constraint has been reached,
  # after the agent's action was applied.
  MAX_STEPS = 1

  # When the agent has been interrupted by the supervisor, due to some
  # internal process, which may or may not be related to agent's action(s).
  INTERRUPTED = 2

  # The episode terminated due to human player exiting the game.
  QUIT = 3
