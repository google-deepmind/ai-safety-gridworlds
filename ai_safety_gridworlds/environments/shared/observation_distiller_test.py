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
"""Tests for pycolab environment initialisations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import absltest

from ai_safety_gridworlds.environments import safe_interruptibility as _safe_interruptibility
from ai_safety_gridworlds.environments.shared import observation_distiller

import numpy as np


class ObservationDistillerTest(absltest.TestCase):

  def testAsciiBoardDistillation(self):
    array_converter = observation_distiller.ObservationToArrayWithRGB(
        value_mapping={'#': 0.0, '.': 0.0, ' ': 1.0,
                       'I': 2.0, 'A': 3.0, 'G': 4.0, 'B': 5.0},
        colour_mapping=_safe_interruptibility.GAME_BG_COLOURS)

    env = _safe_interruptibility.make_game({}, 0, 0.5)
    observations, _, _ = env.its_showtime()
    result = array_converter(observations)

    expected_board = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 4, 0, 0, 0, 3, 0],
         [0, 1, 1, 2, 1, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0]])

    self.assertTrue(np.array_equal(expected_board, result['board']))
    self.assertIn('RGB', list(result.keys()))


if __name__ == '__main__':
  absltest.main()
