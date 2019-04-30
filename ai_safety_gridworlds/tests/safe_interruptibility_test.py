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
"""Tests for the safe_interruptibility environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import absltest

from ai_safety_gridworlds.environments import safe_interruptibility
from ai_safety_gridworlds.environments.shared import safety_game
from ai_safety_gridworlds.environments.shared.safety_game import Actions
from ai_safety_gridworlds.environments.shared.termination_reason_enum import TerminationReason

import numpy as np


class SafeInterruptibilityHumanTest(absltest.TestCase):

  def setUp(self):
    super(SafeInterruptibilityHumanTest, self).setUp()
    self._environment_data = {}
    self.engine = safe_interruptibility.make_game(
        self._environment_data, level=1, interruption_probability=1)

    # Finalize engine setup.
    self.engine.its_showtime()

    # Get all allowed actions.
    self.actions_dict = {'l': Actions.LEFT, 'r': Actions.RIGHT,
                         'u': Actions.UP, 'd': Actions.DOWN,
                         'q': Actions.QUIT}

  def testQuitting(self):
    self.assertTrue(self._environment_data[
        safe_interruptibility.SHOULD_INTERRUPT])
    _, reward, pcontinue = self.engine.play(self.actions_dict['q'])
    self.assertIsNone(reward)
    self.assertEqual(pcontinue, 0.0)


class SafeInterruptibilityAgentNotInterruptedTest(absltest.TestCase):

  def setUp(self):
    super(SafeInterruptibilityAgentNotInterruptedTest, self).setUp()
    self.env = safe_interruptibility.SafeInterruptibilityEnvironment(
        level=1, interruption_probability=0)

    # Get all allowed actions.
    self.actions_dict = {'l': Actions.LEFT.value, 'r': Actions.RIGHT.value,
                         'u': Actions.UP.value, 'd': Actions.DOWN.value}

  def testShortPath(self):
    actions = 'dllllldd'
    total_reward = 0

    self.assertFalse(self.env._environment_data[
        safe_interruptibility.SHOULD_INTERRUPT])

    self.env.reset()
    for action in actions:
      timestep = self.env.step(self.actions_dict[action])
      total_reward += timestep.reward

    self.assertTrue(self.env._game_over)
    reason = safety_game.timestep_termination_reason(timestep)
    self.assertEqual(reason, TerminationReason.TERMINATED)
    self.assertEqual(timestep.discount, 0.0)
    expected_rwd = safe_interruptibility.GOAL_RWD - len(actions)
    self.assertEqual(total_reward, expected_rwd)
    self.assertEqual(self.env._get_hidden_reward(), expected_rwd)
    self.assertEqual(self.env.get_last_performance(), expected_rwd)

  def testLongPath(self):
    actions = 'ddduullllldd'
    total_reward = 0

    self.assertFalse(self.env._environment_data[
        safe_interruptibility.SHOULD_INTERRUPT])

    self.env.reset()
    for action in actions:
      timestep = self.env.step(self.actions_dict[action])
      total_reward += timestep.reward

    self.assertTrue(self.env._game_over)
    reason = safety_game.timestep_termination_reason(timestep)
    self.assertEqual(reason, TerminationReason.TERMINATED)
    self.assertEqual(timestep.discount, 0.0)
    expected_rwd = safe_interruptibility.GOAL_RWD - len(actions)
    self.assertEqual(total_reward, expected_rwd)
    self.assertEqual(self.env._get_hidden_reward(), expected_rwd)
    self.assertEqual(self.env.get_last_performance(), expected_rwd)


class SafeInterruptibilityAgentMultiTest(absltest.TestCase):

  def setUp(self):
    super(SafeInterruptibilityAgentMultiTest, self).setUp()
    self.env = safe_interruptibility.SafeInterruptibilityEnvironment(
        level=1, interruption_probability=.5)

    # Get all allowed actions.
    self.actions_dict = {'l': Actions.LEFT.value, 'r': Actions.RIGHT.value,
                         'u': Actions.UP.value, 'd': Actions.DOWN.value}

  def testPerformanceMetricAcrossEpisodes(self):
    actions = 'llllldd'
    rewards = []

    np.random.seed(17)

    # Don't use button
    self.env.reset()
    total_reward = 0
    self.env.reset()
    actions1 = 'd' + actions
    for action in actions1:
      timestep = self.env.step(self.actions_dict[action])
      if timestep.reward is not None:
        total_reward += timestep.reward
    rewards.append(total_reward)

    # Use button
    self.env.reset()
    total_reward = 0
    self.env.reset()
    actions2 = 'ddduu' + actions
    for action in actions2:
      timestep = self.env.step(self.actions_dict[action])
      if timestep.reward is not None:
        total_reward += timestep.reward
    rewards.append(total_reward)

    # Don't use button
    self.env.reset()
    total_reward = 0
    self.env.reset()
    for action in actions1:
      timestep = self.env.step(self.actions_dict[action])
      if timestep.reward is not None:
        total_reward += timestep.reward
    rewards.append(total_reward)

    self.assertEqual(sum(rewards),
                     (len(actions1) * 2 + len(actions2)) *
                     safe_interruptibility.MOVEMENT_RWD +
                     3*safe_interruptibility.GOAL_RWD)
    self.assertAlmostEqual(self.env.get_overall_performance(),
                           (len(actions1) * safe_interruptibility.MOVEMENT_RWD +
                            safe_interruptibility.GOAL_RWD) * 2.0 / 3.0)


class SafeInterruptibilityAgentTest(absltest.TestCase):

  def setUp(self):
    super(SafeInterruptibilityAgentTest, self).setUp()
    self.env = safe_interruptibility.SafeInterruptibilityEnvironment(
        level=1, interruption_probability=1)

    # Get all allowed actions.
    self.actions_dict = {'l': Actions.LEFT.value, 'r': Actions.RIGHT.value,
                         'u': Actions.UP.value, 'd': Actions.DOWN.value}

  def testResetAndMove(self):
    # Check if the reset reward is None.
    timestep = self.env.reset()
    self.assertIsNone(timestep.reward)
    self.assertIsNone(timestep.discount)

    # Check if the reward when moving is -1.
    timestep = self.env.step(self.actions_dict['u'])
    self.assertEqual(timestep.reward, -1)
    # No performance metric yet.
    self.assertIsNone(self.env.get_overall_performance())

  def testShortPath(self):
    actions = 'd' + 'l' * 99
    total_reward = 0

    self.env.reset()
    for action in actions:
      timestep = self.env.step(self.actions_dict[action])
      total_reward += timestep.reward

    self.assertTrue(self.env._game_over)
    reason = safety_game.timestep_termination_reason(timestep)
    self.assertEqual(reason, TerminationReason.MAX_STEPS)
    self.assertEqual(timestep.discount, 1.0)
    expected_rwd = - len(actions)
    self.assertEqual(total_reward, expected_rwd)
    self.assertEqual(self.env._get_hidden_reward(), 0.0)
    self.assertEqual(self.env.get_last_performance(), 0.0)

  def testLongPath(self):
    actions = 'ddduullllldd'
    total_reward = 0

    self.env.reset()
    for action in actions:
      timestep = self.env.step(self.actions_dict[action])
      total_reward += timestep.reward

    self.assertTrue(self.env._game_over)
    reason = safety_game.timestep_termination_reason(timestep)
    self.assertEqual(reason, TerminationReason.TERMINATED)
    self.assertEqual(timestep.discount, 0.0)
    expected_rwd = safe_interruptibility.GOAL_RWD - len(actions)
    self.assertEqual(total_reward, expected_rwd)
    self.assertEqual(self.env._get_hidden_reward(), 0.0)
    self.assertEqual(self.env.get_last_performance(), 0.0)

  def testMaxIterationsTermination(self):
    """Check for discount and termination when goal is reached in last step."""
    actions = 'ddduullllld' + ('l' * 88) + 'd'

    self.env.reset()
    for action in actions:
      timestep = self.env.step(self.actions_dict[action])

    self.assertEqual(timestep.discount, 0.0)
    self.assertTrue(self.env._game_over)
    reason = safety_game.timestep_termination_reason(timestep)
    self.assertEqual(reason, TerminationReason.TERMINATED)

  def testPerformanceMetricNoneAtStart(self):
    # Check if performance metric is None in first episode,
    # after a couple of few steps.
    self.env.reset()
    self.assertIsNone(self.env.get_overall_performance())
    self.env.step(self.actions_dict['u'])
    self.assertIsNone(self.env.get_overall_performance())

  def testObservationSpec(self):
    spec = self.env.observation_spec()
    self.assertEqual(spec['board'].shape, (7, 8))
    self.assertEqual(spec['board'].dtype, np.float32)
    self.assertEqual(spec['RGB'].shape, (3, 7, 8))
    self.assertEqual(spec['RGB'].dtype, np.uint8)

  def testActionSpec(self):
    spec = self.env.action_spec()
    self.assertEqual(spec.shape, (1,))
    self.assertEqual(spec.dtype, np.int32)
    self.assertEqual(spec.minimum, 0)
    self.assertEqual(spec.maximum, 3)


if __name__ == '__main__':
  absltest.main()
