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
"""Tests for `demonstrations.py`.

This set of tests will execute the demonstrations by constructing an instance
of each of the environments and running the sequence of actions, checking
return and safety performance.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized

from ai_safety_gridworlds.demonstrations import demonstrations
from ai_safety_gridworlds.environments.shared import safety_game
from ai_safety_gridworlds.environments.shared.termination_reason_enum import TerminationReason
from ai_safety_gridworlds.helpers import factory

import numpy as np
from six.moves import range


REPETITIONS = 10  # How many times to run each demonstration.


class DemonstrationsTest(parameterized.TestCase):
  """Test demonstrations by replaying them in their environment."""

  def _run_env(self, env, action_sequence, terminated=True):
    """Execute an action sequence in an environment."""
    env.reset()
    episode_return = 0
    for action in action_sequence:
      timestep = env.step(action)
      if timestep.reward is not None:
        episode_return += timestep.reward

    # Check the termination condition
    interruption = (env.environment_data.get(
        safety_game.TERMINATION_REASON, False) == TerminationReason.INTERRUPTED)
    max_steps = (env.environment_data.get(
        safety_game.TERMINATION_REASON, False) == TerminationReason.MAX_STEPS)

    self.assertEqual(timestep.discount == 0.0 or interruption or max_steps,
                     terminated)
    return episode_return

  def test_not_available(self):
    """Test that using an unavailable environment name raises a `ValueError`."""
    unavailable = 'ksljadflkwjeflinsdflkskldjfkldf'  # something obscure
    self.assertRaises(ValueError, demonstrations.get_demonstrations,
                      unavailable)

  @parameterized.named_parameters(
      *[('_' + name, name) for name in demonstrations.environment_names()]
  )
  def test_demonstrations(self, environment_name):
    """Execute the demonstrations in the given environment."""
    demos = demonstrations.get_demonstrations(environment_name)

    # Execute each demonstration.
    for demo in demos:
      # Run several times to be sure that result is deterministic.
      for _ in range(REPETITIONS):
        # Fix random seed.
        np.random.seed(demo.seed)

        # Construct and run environment.
        env = factory.get_environment_obj(environment_name)
        episode_return = self._run_env(env, demo.actions, demo.terminates)

        # Check return and safety performance.
        self.assertEqual(episode_return, demo.episode_return)
        if demo.terminates:
          hidden_reward = env.get_overall_performance()
        else:
          hidden_reward = env._get_hidden_reward(default_reward=None)
        if hidden_reward is not None:
          self.assertEqual(hidden_reward, demo.safety_performance)


if __name__ == '__main__':
  absltest.main()
