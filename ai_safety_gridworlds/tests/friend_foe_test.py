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

"""Tests for Friend or Foe environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from ai_safety_gridworlds.environments import friend_foe
from ai_safety_gridworlds.environments.shared.safety_game import Actions
from pycolab.tests import test_things as tt


class FriendFoeHumanTest(absltest.TestCase):

  def setUp(self):
    # Get all allowed actions.
    self.actions_dict = {'l': Actions.LEFT, 'r': Actions.RIGHT,
                         'u': Actions.UP, 'd': Actions.DOWN,
                         'q': Actions.QUIT}

  def testQuitting(self):
    self.engine = friend_foe.make_game({})
    # Finalize engine setup.
    self.engine.its_showtime()

    _, reward, pcontinue = self.engine.play(self.actions_dict['q'])
    self.assertEqual(reward, None)
    self.assertEqual(pcontinue, 0.0)


class FriendFoeAgentTest(parameterized.TestCase,
                         tt.PycolabTestCase):

  def setUp(self):
    # Get all allowed actions.
    self.actions_dict = {'l': Actions.LEFT, 'r': Actions.RIGHT,
                         'u': Actions.UP, 'd': Actions.DOWN}

  @parameterized.named_parameters(
      ('_Left', 'uuul'),
      ('_Right', 'uuur'),
  )
  def testStepIntoGoal_Terminate(self, actions):
    self.env = friend_foe.FriendFoeEnvironment(bandit_type='adversary')
    self.env.reset()

    for action in actions:
      step = self.env.step(self.actions_dict[action])

    self.assertEqual(step.discount, 0.0)

  @parameterized.named_parameters(
      ('_Left', 'uuul', np.array([14, 35])),
      ('_Right', 'uuur', np.array([9, 33]))
  )
  def testMeanRewards(self, actions, band):
    np.random.seed(2)
    self.env = friend_foe.FriendFoeEnvironment()

    # Initialize arrays for estimating mean rewards.
    cumulative_rewards = np.zeros(3)
    cumulative_actions = np.zeros(3)

    # Run episodes and accumulate final rewards and actions per bandit type.
    for _ in range(1000):
      self.env.reset()
      for action in actions:
        step = self.env.step(self.actions_dict[action])
        bandit_type = self.env.environment_data['current_episode_bandit']
      cumulative_rewards[bandit_type] += step.reward
      cumulative_actions[bandit_type] += 1.0

    # Calculate mean rewards.
    mean_rewards = cumulative_rewards / cumulative_actions

    # Test whether friendly, neutral, and adversarial bandits
    # yield mean rewards that fall between the specified bands.
    self.assertTrue(band[1] < mean_rewards[0])
    self.assertTrue(band[0] < mean_rewards[1] and mean_rewards[1] < band[1])
    self.assertTrue(mean_rewards[2] < band[0])

  def testRevealGoal(self):
    np.random.seed(0)

    game_art = ['#1#0#',
                '#AB*#',
                '#BBB#',
                '#BBB#',
                '#BBB#',
                '#####']

    env = friend_foe.FriendFoeEnvironment(bandit_type='adversary',
                                          extra_step=True)
    actions = 'uuuul'
    for action in actions:
      step = env.step(self.actions_dict[action])
    self.assertEqual(step.discount, 1.0)
    self.assertBoard(env.current_game._board.board, game_art)

  def testObservationSpec(self):
    self.env = friend_foe.FriendFoeEnvironment()
    spec = self.env.observation_spec()
    self.assertEqual(spec['board'].shape, (6, 5))
    self.assertEqual(spec['board'].dtype, np.float32)
    self.assertEqual(spec['RGB'].shape, (3, 6, 5))
    self.assertEqual(spec['RGB'].dtype, np.uint8)

  def testActionSpec(self):
    self.env = friend_foe.FriendFoeEnvironment()
    spec = self.env.action_spec()
    self.assertEqual(spec.shape, (1,))
    self.assertEqual(spec.dtype, np.int32)
    self.assertEqual(spec.minimum, 0)
    self.assertEqual(spec.maximum, 3)


if __name__ == '__main__':
  absltest.main()
