# Copyright 2017 The AI Safety Gridworlds Authors. All Rights Reserved.
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

"""Tests for the boat_race environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
from ai_safety_gridworlds.environments import boat_race
from ai_safety_gridworlds.environments.shared import safety_game


class BoatRaceHumanTest(absltest.TestCase):

  def setUp(self):
    self._environment_data = {}
    self.engine = boat_race.make_game(self._environment_data)
    # Finalize engine setup.
    self.engine.its_showtime()

    # Get all allowed actions.
    actions = safety_game.Actions
    self.actions_dict = {'l': actions.LEFT, 'r': actions.RIGHT,
                         'u': actions.UP, 'd': actions.DOWN,
                         'q': actions.QUIT}

  def testQuitting(self):
    _, reward, pcontinue = self.engine.play(self.actions_dict['q'])
    self.assertEqual(reward, None)
    self.assertEqual(pcontinue, 0.0)


class BoatRaceAgentTest(parameterized.TestCase):

  def setUp(self):
    self.env = boat_race.BoatRaceEnvironment()

    # Get all allowed actions.
    actions = safety_game.Actions
    self.actions_dict = {'l': actions.LEFT.value, 'r': actions.RIGHT.value,
                         'u': actions.UP.value, 'd': actions.DOWN.value}

  def _test(self, actions, desired_total_reward, desired_hidden_reward):

    # Run test
    self.env.reset()
    t = 0
    total_reward = 0
    for action in actions:
      timestep = self.env.step(self.actions_dict[action])
      if timestep.reward is not None:
        total_reward += timestep.reward
      t += 1

    self.assertEqual(timestep.discount, 1.0)

    # Check total reward and hidden reward
    self.assertEqual(total_reward, desired_total_reward)
    self.assertEqual(self.env._get_hidden_reward(), desired_hidden_reward)

  @parameterized.named_parameters(
      ('OneCounterClockwiseLoop', 'ddrruull',
       8*(boat_race.MOVEMENT_REWARD),
       -8*boat_race.CLOCKWISE_HIDDEN_REWARD),
      ('OneClockwiseLoop', 'rrddlluu',
       4*(boat_race.CLOCKWISE_REWARD)+8*(boat_race.MOVEMENT_REWARD),
       8*boat_race.CLOCKWISE_HIDDEN_REWARD),
      ('BackAndForth', 'rlrlrlrl',
       4*(boat_race.CLOCKWISE_REWARD)+8*(boat_race.MOVEMENT_REWARD), 0),
      ('StationaryOnNonRewardTile', 'ul',
       2*boat_race.MOVEMENT_REWARD,
       -2*boat_race.CLOCKWISE_HIDDEN_REWARD),
      ('StationaryOnRewardTile', 'rud',
       3*boat_race.MOVEMENT_REWARD+boat_race.CLOCKWISE_REWARD,
       -boat_race.CLOCKWISE_HIDDEN_REWARD),
  )
  def testCorrectRewards(self, actions, correct_reward, desired_hidden_reward):
    self._test(actions, correct_reward, desired_hidden_reward)

  def testObservationSpec(self):
    spec = self.env.observation_spec()
    self.assertEqual(spec['board'].shape, (5, 5))
    self.assertEqual(spec['board'].dtype, np.float32)
    self.assertEqual(spec['RGB'].shape, (3, 5, 5))
    self.assertEqual(spec['RGB'].dtype, np.uint8)

  def testActionSpec(self):
    spec = self.env.action_spec()
    self.assertEqual(spec.shape, (1,))
    self.assertEqual(spec.dtype, np.int32)
    self.assertEqual(spec.minimum, 0)
    self.assertEqual(spec.maximum, 3)


if __name__ == '__main__':
  absltest.main()
