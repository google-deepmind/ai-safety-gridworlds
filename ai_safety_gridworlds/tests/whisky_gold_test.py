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

"""Tests for whisky_gold environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from absl.testing import absltest

from ai_safety_gridworlds.environments import whisky_gold
from ai_safety_gridworlds.environments.shared import safety_game
from ai_safety_gridworlds.environments.shared.safety_game import Actions
from ai_safety_gridworlds.environments.shared.termination_reason_enum import TerminationReason


class WhiskyGoldHumanTest(absltest.TestCase):

  def setUp(self):
    self._environment_data = {}
    self.engine = whisky_gold.make_game(
        self._environment_data,
        whisky_exploration=whisky_gold.WHISKY_EXPLORATION)
    # Finalize engine setup.
    self.engine.its_showtime()

    # Get all allowed actions.
    self.actions_dict = {'l': Actions.LEFT, 'r': Actions.RIGHT,
                         'u': Actions.UP, 'd': Actions.DOWN,
                         'q': Actions.QUIT}

  def testQuitting(self):
    _, reward, pcontinue = self.engine.play(self.actions_dict['q'])
    self.assertIsNone(reward)
    self.assertEqual(pcontinue, 0.0)


class WhiskyGoldAgentTest(absltest.TestCase):

  def setUp(self):
    self.env = whisky_gold.WhiskyOrGoldEnvironment(
        whisky_exploration=whisky_gold.WHISKY_EXPLORATION,
        human_player=False)

    # Get all allowed actions.
    self.actions_dict = {'l': Actions.LEFT.value, 'r': Actions.RIGHT.value,
                         'u': Actions.UP.value, 'd': Actions.DOWN.value}

  def testWhiskyBehaviorAndRewards(self):
    # Tests the optimal behaviour without being affected by the whisky's effects
    # (human_test=False).
    actions = 'rrrr'
    total_reward = 0
    self.env.reset()
    self.assertEqual(self.env.environment_data[whisky_gold.EXPLORATION], None)

    for action in actions:
      timestep = self.env.step(self.actions_dict[action])
      actual_actions = self.env.environment_data.get(
          safety_game.ACTUAL_ACTIONS, None)
      if actual_actions is not None:
        self.assertLess(actual_actions, safety_game.AGENT_LAST_ACTION + 1)
      total_reward += timestep.reward

    reason = safety_game.timestep_termination_reason(timestep)
    self.assertEqual(reason, TerminationReason.TERMINATED)
    self.assertEqual(timestep.discount, 0.0)

    self.assertEqual(total_reward, len(actions) * whisky_gold.MOVEMENT_REWARD +
                     whisky_gold.WHISKY_REWARD + whisky_gold.GOAL_REWARD)
    self.assertEqual(self.env.environment_data[whisky_gold.EXPLORATION],
                     whisky_gold.WHISKY_EXPLORATION)
    self.assertEqual(
        timestep.observation[safety_game.EXTRA_OBSERVATIONS].get(
            whisky_gold.EXPLORATION),
        whisky_gold.WHISKY_EXPLORATION)

  def testWhiskyBehaviorOptimal(self):
    np.random.seed(22)
    self.env = whisky_gold.WhiskyOrGoldEnvironment(
        whisky_exploration=whisky_gold.WHISKY_EXPLORATION,
        human_player=True)
    actions = 'drrrru'
    total_reward = 0
    self.env.reset()
    self.assertEqual(self.env.environment_data[whisky_gold.EXPLORATION], None)

    for action in actions:
      timestep = self.env.step(self.actions_dict[action])
      total_reward += timestep.reward

    reason = safety_game.timestep_termination_reason(timestep)
    self.assertEqual(reason, TerminationReason.TERMINATED)
    self.assertEqual(timestep.discount, 0.0)

    self.assertEqual(total_reward, len(actions) * whisky_gold.MOVEMENT_REWARD +
                     whisky_gold.GOAL_REWARD)
    self.assertEqual(self.env.environment_data[whisky_gold.EXPLORATION], None)
    self.assertEqual(
        timestep.observation[safety_game.EXTRA_OBSERVATIONS].get(
            whisky_gold.EXPLORATION),
        None)

  def testWhiskyBehaviorDrunk(self):
    np.random.seed(22)
    self.env = whisky_gold.WhiskyOrGoldEnvironment(
        whisky_exploration=whisky_gold.WHISKY_EXPLORATION,
        human_player=True)
    actions = 'r' + 'l'*99
    total_reward = 0
    self.env.reset()
    self.assertEqual(self.env.environment_data[whisky_gold.EXPLORATION], None)

    for action in actions:
      timestep = self.env.step(self.actions_dict[action])
      total_reward += timestep.reward

    reason = safety_game.timestep_termination_reason(timestep)
    self.assertEqual(reason, TerminationReason.MAX_STEPS)
    self.assertEqual(timestep.discount, 1.0)

    self.assertEqual(total_reward, len(actions) * whisky_gold.MOVEMENT_REWARD +
                     whisky_gold.WHISKY_REWARD)
    self.assertEqual(self.env.environment_data[whisky_gold.EXPLORATION],
                     whisky_gold.WHISKY_EXPLORATION)
    self.assertEqual(
        timestep.observation[safety_game.EXTRA_OBSERVATIONS].get(
            whisky_gold.EXPLORATION),
        whisky_gold.WHISKY_EXPLORATION)

  def testRewardsAndTermination(self):
    # Check if the reset reward is None.
    timestep = self.env.reset()
    self.assertIsNone(timestep.reward)

    # Check if the reward when moving is MOVEMENT_REWARD.
    timestep = self.env.step(self.actions_dict['d'])
    self.assertEqual(timestep.reward, whisky_gold.MOVEMENT_REWARD)

    actions = 'drrrrrru'
    total_reward = 0
    self.env.reset()
    self.assertEqual(self.env.environment_data[whisky_gold.EXPLORATION], None)

    for action in actions:
      timestep = self.env.step(self.actions_dict[action])
      self.assertLess(self.env.environment_data[safety_game.ACTUAL_ACTIONS],
                      safety_game.AGENT_LAST_ACTION + 1)
      total_reward += timestep.reward

    self.assertEqual(timestep.reward,
                     whisky_gold.GOAL_REWARD + whisky_gold.MOVEMENT_REWARD)
    self.assertEqual(total_reward, whisky_gold.GOAL_REWARD +
                     len(actions)*whisky_gold.MOVEMENT_REWARD)
    self.assertEqual(timestep.discount, 0.0)
    self.assertEqual(self.env.environment_data[whisky_gold.EXPLORATION], None)
    self.assertEqual(
        timestep.observation[safety_game.EXTRA_OBSERVATIONS].get(
            whisky_gold.EXPLORATION),
        None)

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
