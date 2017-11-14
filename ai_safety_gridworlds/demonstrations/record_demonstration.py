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

"""Records a new demonstration using the commandline.

Use for example like this:

    $ blaze build :record_demonstration
    $ bb record_demonstration --environment=safe_interruptibility

See `bb record_demonstration --help` for more command line options.

Note: if the environment doesn't terminate upon your action sequence, you can
use `quit` action to terminate it yourself and this will not be recorded in the
output sequence.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import numpy as np

from absl import app
from absl import flags

from ai_safety_gridworlds.demonstrations import demonstrations
from ai_safety_gridworlds.environments.shared import safety_ui
from ai_safety_gridworlds.helpers import factory


FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', None, 'Random seed for the environment.')
flags.DEFINE_string('environment', None, 'Name of the environment.')
flags.mark_flag_as_required('environment')


def _postprocess_actions(actions_list):
  to_char = {a: c for c, a in demonstrations._actions.items()}  # pylint: disable=protected-access
  actions = [to_char[a] for a in actions_list if a is not None]
  return ''.join(actions)


def main(unused_argv):
  # Set random seed.
  if FLAGS.seed is not None:
    seed = FLAGS.seed
  else:
    # Get a new random random seed and remember it.
    seed = np.random.randint(0, 100)
  np.random.seed(seed)

  # Run one episode.
  actions_list = []  # This stores the actions taken.
  env = factory.get_environment_obj(FLAGS.environment)
  # Get the module so we can obtain environment specific constants.
  module = importlib.import_module(env.__class__.__module__)

  # Overwrite the environment's step function to record the actions.
  old_step = env.step
  def _step(actions):
    actions_list.append(actions)
    return old_step(actions)
  env.step = _step
  ui = safety_ui.make_human_curses_ui(module.GAME_BG_COLOURS,
                                      module.GAME_FG_COLOURS)
  ui.play(env)

  # Extract data
  episode_return = env.episode_return
  safety_performance = env.get_overall_performance()
  actions = _postprocess_actions(actions_list)

  # Determine termination reason.
  if actions[-1] == 'q':
    # Player has quit the game, remove it from the sequence.
    actions = actions[:-1]
    terminates = False
  else:
    terminates = True

  # Print the resulting demonstration to the terminal.
  demo = demonstrations.Demonstration(seed, actions, episode_return,
                                      safety_performance, terminates)
  print('Recorded the following data:\n{}'.format(demo))


if __name__ == '__main__':
  app.run(main)
