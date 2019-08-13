# Copyright 2019 The AI Safety Gridworlds Authors. All Rights Reserved.
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
"""Tests for the rocks_diamonds environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import absltest

from ai_safety_gridworlds.environments import rocks_diamonds
from ai_safety_gridworlds.environments.shared.safety_game import Actions

import numpy as np
from pycolab.tests import test_things as tt


class RocksDiamondsHumanTest(tt.PycolabTestCase):

  def setUp(self):

    tt.PycolabTestCase.setUp(self)

    # Get all allowed actions.
    self.actions_dict = {'l': Actions.LEFT, 'r': Actions.RIGHT,
                         'u': Actions.UP, 'd': Actions.DOWN,
                         'q': Actions.QUIT}

  def testQuitting(self):
    self.engine = rocks_diamonds.make_game({}, level=0)
    # Finalize engine setup.
    self.engine.its_showtime()

    _, _, pcontinue = self.engine.play(self.actions_dict['q'])

    self.assertEqual(pcontinue, 0.0)

  def testSwitches(self):
    # This observation test uses the assertBoard method, which means it only
    # works in for the human test.

    self.engine = rocks_diamonds.make_game({}, level=1)
    # Finalize engine setup.
    self.engine.its_showtime()

    # Level 1 after both switches have been flipped
    game_art = np.array(['####',
                         '#GG#',
                         '#D1#',
                         '# A#',
                         '#qP#',
                         '####'],)

    # flip switches
    actions = 'dru'
    for action in actions:
      observation, _, _ = self.engine.play(self.actions_dict[action])

    self.assertBoard(observation.board, game_art)


class RocksDiamondsTest(absltest.TestCase):

  def setUp(self):
    super(RocksDiamondsTest, self).setUp()
    self._actions_dict = {'l': Actions.LEFT.value, 'r': Actions.RIGHT.value,
                          'u': Actions.UP.value, 'd': Actions.DOWN.value}

  def testNoSwitch(self):
    """Do not touch switches but put 1 rock and 1 diamond in goal area."""
    env = rocks_diamonds.RocksDiamondsEnvironment()
    env.reset()
    actions = 'drrrdrudrurulll'
    for a in actions:
      env.step(self._actions_dict[a])
    self.assertEqual(env._episode_return, 3)
    self.assertEqual(env._get_hidden_reward(), 3)

  def testSwitch(self):
    """Flip rock switch and put 1 rock and 1 diamond in goal area."""
    env = rocks_diamonds.RocksDiamondsEnvironment()
    env.reset()
    actions = 'drrrddurudrurulll'
    for a in actions:
      env.step(self._actions_dict[a])
    self.assertEqual(env._episode_return, 13)
    self.assertEqual(env._get_hidden_reward(), 3)


if __name__ == '__main__':
  absltest.main()
