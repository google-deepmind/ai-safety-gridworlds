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
"""Tests for distributional_shift environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import absltest
from absl.testing import parameterized

from ai_safety_gridworlds.environments import distributional_shift
from ai_safety_gridworlds.environments.shared.safety_game import Actions


class DistributionalShiftHumanTest(absltest.TestCase):

  def setUp(self):
    super(DistributionalShiftHumanTest, self).setUp()
    # Get all allowed actions.
    self.actions_dict = {'l': Actions.LEFT, 'r': Actions.RIGHT,
                         'u': Actions.UP, 'd': Actions.DOWN,
                         'q': Actions.QUIT}

  def testQuitting(self):
    self.engine = distributional_shift.make_game({}, is_testing=False)
    # Finalize engine setup.
    self.engine.its_showtime()

    _, reward, pcontinue = self.engine.play(self.actions_dict['q'])
    self.assertIsNone(reward)
    self.assertEqual(pcontinue, 0.0)


class DistributionalShiftAgentTrainingTest(absltest.TestCase):

  def setUp(self):
    super(DistributionalShiftAgentTrainingTest, self).setUp()
    self.env = distributional_shift.DistributionalShiftEnvironment(False)

    # Get all allowed actions.
    self.actions_dict = {'l': Actions.LEFT.value, 'r': Actions.RIGHT.value,
                         'u': Actions.UP.value, 'd': Actions.DOWN.value}

  def testGoalAndTermination(self):
    actions = 'drrrrrru'
    total_reward = 0

    self.env.reset()
    for action in actions:
      timestep = self.env.step(self.actions_dict[action])
      total_reward += timestep.reward

    final_reward = (distributional_shift.GOAL_REWARD +
                    distributional_shift.MOVEMENT_REWARD)
    final_total_reward = (distributional_shift.GOAL_REWARD +
                          len(actions) * distributional_shift.MOVEMENT_REWARD)
    self.assertEqual(timestep.reward, final_reward)
    self.assertEqual(total_reward, final_total_reward)
    self.assertEqual(timestep.discount, 0.0)

  def testLavaAndTermination(self):
    actions = 'rr'
    total_reward = 0

    self.env.reset()
    for action in actions:
      timestep = self.env.step(self.actions_dict[action])
      total_reward += timestep.reward

    final_reward = (distributional_shift.LAVA_REWARD +
                    distributional_shift.MOVEMENT_REWARD)
    final_total_reward = (distributional_shift.LAVA_REWARD +
                          len(actions) * distributional_shift.MOVEMENT_REWARD)
    self.assertEqual(timestep.reward, final_reward)
    self.assertEqual(total_reward, final_total_reward)
    self.assertEqual(timestep.discount, 0.0)

  def testMapShape(self):
    timestep = self.env.reset()
    lava_top = timestep.observation['board'][1][3:6]
    lava_bottom = timestep.observation['board'][-2][3:6]
    self.assertTrue((lava_top == 4.0).all())
    self.assertTrue((lava_bottom == 4.0).all())


class DistributionalShiftAgentTestingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('TopShift', 1, (1, 3)),
      ('BottomShift', 2, (-2, -3)),
  )
  def testMapShape(self, level, rows):
    self.env = distributional_shift.DistributionalShiftEnvironment(
        is_testing=True, level_choice=level)
    timestep = self.env.reset()
    lava = timestep.observation['board'][rows[0]:rows[1], 3:6]
    self.assertTrue((lava == 4.0).all())


if __name__ == '__main__':
  absltest.main()
