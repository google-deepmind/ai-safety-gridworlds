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
"""Frontends for humans who want to play pycolab games."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses

from ai_safety_gridworlds.environments.shared import safety_game
from ai_safety_gridworlds.environments.shared import safety_ui

import numpy as np


# adapted from ai_safety_gridworlds\environments\shared\safety_ui.py
class SafetyCursesUiEx(safety_ui.SafetyCursesUi):
  """A terminal-based UI for pycolab games.

  This is deriving from pycolab's `human_ui.CursesUi` class and shares a
  lot of its code. The main purpose of having a separate class is that we want
  to use the `play()` method on an instance of `SafetyEnvironment` and not just
  a pycolab game `Engine`. This way we can store information across
  episodes, conveniently call `get_overall_performance()` after the human has
  finished playing. It is also ensuring that human and agent interact with the
  environment in the same way (e.g. if `SafetyEnvironment` gets derived).
  """

  #def __init__(self, *args, **kwargs):
  #  super(SafetyCursesUiEx, self).__init__(*args, **kwargs)
  #  self.prev_metrics = None

  def _display(self, screen, *args, **kwargs):
    super(SafetyCursesUiEx, self)._display(screen, *args, **kwargs)

    metrics = self._env._environment_data.get("metrics")
    if metrics is not None:

      # print(metrics)

      padding = 2
      cell_widths = [padding + max(len(str(cell)) for cell in col) for col in metrics.T]

      #prev_metrics = self.prev_metrics if self.prev_metrics is not None else np.empty([0, 0])

      for row_index, row in enumerate(metrics):
        for col_index, cell in enumerate(row):

          #if row_index < prev_metrics.shape[0] and col_index < prev_metrics.shape[1]:
          #  prev_metric = prev_metrics[row_index, col_index]
          #else:
          #  prev_metric = ""

          #if prev_metric != cell:  # reduce flicker
          screen.addstr(2 + row_index, 20 + (cell_widths[col_index - 1] if col_index > 0 else 0), str(cell), curses.color_pair(0)) 
            
      #self.prev_metrics = metrics.copy()


# adapted from ai_safety_gridworlds\environments\shared\safety_ui.py
def make_human_curses_ui_with_noop_keys(game_bg_colours, game_fg_colours, noop_keys, delay=100):
  """Instantiate a Python Curses UI for the terminal game.

  Args:
    game_bg_colours: dict of game element background colours.
    game_fg_colours: dict of game element foreground colours.
    noop_keys: enables NOOP actions on keyboard using space bar and middle button on keypad.
    delay: in ms, how long does curses wait before emitting a noop action if
      such an action exists. If it doesn't it just waits, so this delay has no
      effect. Our situation is the latter case, as we don't have a noop.

  Returns:
    A curses UI game object.
  """

  keys_to_actions={curses.KEY_UP:       safety_game.Actions.UP,
                    curses.KEY_DOWN:    safety_game.Actions.DOWN,
                    curses.KEY_LEFT:    safety_game.Actions.LEFT,
                    curses.KEY_RIGHT:   safety_game.Actions.RIGHT,
                    'q':                safety_game.Actions.QUIT,
                    'Q':                safety_game.Actions.QUIT}
  if noop_keys:
     keys_to_actions.update({
        # middle button on keypad
        curses.KEY_B2: safety_game.Actions.NOOP,  # KEY_B2: Center of keypad - https://docs.python.org/3/library/curses.html
        # space bar
        ' ': safety_game.Actions.NOOP,
        # -1: Actions.NOOP,           # curses delay timeout "keycode" is -1
      })

  return SafetyCursesUiEx(  
      keys_to_actions=keys_to_actions,
      delay=delay,
      repainter=None,   # TODO
      colour_fg=game_fg_colours,
      colour_bg=game_bg_colours)


