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
"""Tests for the island navigation environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import absltest
from absl.testing import parameterized

from ai_safety_gridworlds.environments import island_navigation
from ai_safety_gridworlds.environments.shared.safety_game import Actions

import numpy as np


class IslandNavigationHumanTest(absltest.TestCase):

  def testQuitting(self):
    self.engine = island_navigation.make_game({})
    # Finalize engine setup.
    self.engine.its_showtime()
    _, _, pcontinue = self.engine.play(Actions.QUIT)
    self.assertEqual(pcontinue, 0.0)


class IslandNavigationEnvironmentTest(parameterized.TestCase):

  def setUp(self):
    super(IslandNavigationEnvironmentTest, self).setUp()
    self.env = island_navigation.IslandNavigationEnvironment()

    # Get all allowed actions.
    self.actions_dict = {'l': Actions.LEFT.value, 'r': Actions.RIGHT.value,
                         'u': Actions.UP.value, 'd': Actions.DOWN.value}

  def _test(self, actions, desired_total_reward, terminated=True):
    # Run test
    self.env.reset()
    t = 0
    total_reward = 0
    timestep = None
    for action in actions:
      timestep = self.env.step(self.actions_dict[action])
      if timestep.reward is not None:
        total_reward += timestep.reward
      t += 1

    # Check the termination condition
    if timestep:
      self.assertEqual(timestep.discount == 0.0, terminated)

    # Check total reward and episode length
    self.assertEqual(total_reward, desired_total_reward)
    self.assertLen(actions, t)

  def testMovementReward(self):
    # Check if the reset reward is None.
    timestep = self.env.reset()
    self.assertIsNone(timestep.reward)
    self.assertIsNone(timestep.discount)

    # Check the movement reward.
    timestep = self.env.step(self.actions_dict['d'])
    self.assertEqual(timestep.reward, island_navigation.MOVEMENT_REWARD)

  @parameterized.named_parameters(
      ('_water_right', 'rrr', -3, -53),
      ('_water_left', 'lll', -3, -53),
      ('_water_down', 'dddrr', -5, -55),
      ('_goal1', 'dddl', 46, 46),
      ('_goal2', 'lddd', 46, 46),
      ('_goal_long_path', 'rrddldll', 42, 42)
  )
  def test(self, actions, episode_return, performance):
    """Test various action sequences that end the episode."""
    self._test(actions, episode_return)
    self.assertEqual(self.env._get_hidden_reward(), performance)

  @parameterized.named_parameters(
      ('1', '', 0, 3),
      ('2', 'l', -1, 2),
      ('3', 'll', -2, 1),
      ('4', 'ddd', -3, 2)
  )
  def testSafetyConstraint(self, actions, rewards, safety):
    self._test(actions, rewards, terminated=False)
    self.assertEqual(self.env.environment_data['safety'], safety)

  def testObservationSpec(self):
    spec = self.env.observation_spec()
    self.assertEqual(spec['board'].shape, (6, 8))
    self.assertEqual(spec['board'].dtype, np.float32)
    self.assertEqual(spec['RGB'].shape, (3, 6, 8))
    self.assertEqual(spec['RGB'].dtype, np.uint8)

  def testActionSpec(self):
    spec = self.env.action_spec()
    self.assertEqual(spec.shape, (1,))
    self.assertEqual(spec.dtype, np.int32)
    self.assertEqual(spec.minimum, 0)
    self.assertEqual(spec.maximum, 3)

if __name__ == '__main__':
  absltest.main()
