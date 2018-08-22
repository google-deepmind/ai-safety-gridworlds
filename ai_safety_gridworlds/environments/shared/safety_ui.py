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
import datetime
import sys

# Dependency imports
from absl import flags

from ai_safety_gridworlds.environments.shared import safety_game
from ai_safety_gridworlds.environments.shared.safety_game import Actions

from pycolab import human_ui
from pycolab.protocols import logging as plab_logging

import six


FLAGS = flags.FLAGS
flags.DEFINE_bool('eval', False, 'Which type of information to print.')
# The launch_human_eval_env.sh can launch environments with --eval, which makes
# score, safety_performance, and environment_data to be printed to stderr for
# easy piping to a separate file.
# The flag --eval also prevents the safety_performance to printed to stdout.


class SafetyCursesUi(human_ui.CursesUi):
  """A terminal-based UI for pycolab games.

  This is deriving from pycolab's `human_ui.CursesUi` class and shares a
  lot of its code. The main purpose of having a separate class is that we want
  to use the `play()` method on an instance of `SafetyEnvironment` and not just
  a pycolab game `Engine`. This way we can store information across
  episodes, conveniently call `get_overall_performance()` after the human has
  finished playing. It is also ensuring that human and agent interact with the
  environment in the same way (e.g. if `SafetyEnvironment` gets derived).
  """

  def __init__(self, *args, **kwargs):
    super(SafetyCursesUi, self).__init__(*args, **kwargs)
    self._env = None

  def play(self, env):
    """Play a pycolab game.

    Calling this method initialises curses and starts an interaction loop. The
    loop continues until the game terminates or an error occurs.

    This method will exit cleanly if an exception is raised within the game;
    that is, you shouldn't have to reset your terminal.

    Args:
      env: An instance of `SafetyEnvironment`.

    Raises:
      RuntimeError: if this method is called while a game is already underway.
      ValueError: if `env` is not an instance of `SafetyEnvironment`.
    """
    if not isinstance(env, safety_game.SafetyEnvironment):
      raise ValueError('`env` must be an instance of `SafetyEnvironment`.')
    if self._game is not None:
      raise RuntimeError('CursesUi is not at all thread safe')
    self._env = env
    self._game = None
    self._start_time = datetime.datetime.now()

    # Inform the environment its playing through curses
    self._env.environment_data[safety_game.CURSES] = True

    # After turning on curses, set it up and play the game.
    curses.wrapper(self._init_curses_and_play)

    # The game has concluded. Print the final statistics.
    score = self._env.episode_return
    duration = datetime.datetime.now() - self._start_time
    termination_reason = env.environment_data[safety_game.TERMINATION_REASON]
    safety_performance = self._env.get_overall_performance()
    print('Game over! Final score is {}, earned over {}.'.format(
        score, _format_timedelta(duration)))
    print('Termination reason: {!s}'.format(termination_reason))

    # If running in eval mode, print results to stderr for piping to file,
    # otherwise print safety performance to user.
    if FLAGS.eval:
      datastr = (
          'score: {}, '
          'safety_performance: {}, '
          'termination_reason: {!s}, '
          'date: {}, '
          'environment_data: {}'
      ).format(
          score,
          safety_performance,
          termination_reason,
          six.text_type(datetime.datetime.utcnow()),
          env.environment_data
      )
      print('{' + datastr + '}', file=sys.stderr)
    else:
      if safety_performance is not None:
        print('Safety performance is {}.'.format(safety_performance))

    # Clean up in preparation for the next game.
    self._game = None
    self._start_time = None

  def _init_curses_and_play(self, screen):
    """Set up an already-running curses; do interaction loop.

    This method is intended to be passed as an argument to `curses.wrapper`,
    so its only argument is the main, full-screen curses window.

    Args:
      screen: the main, full-screen curses window.

    Raises:
      ValueError: if any key in the `keys_to_actions` dict supplied to the
          constructor has already been reserved for use by `CursesUi`.
    """
    # This needs to be overwritten to use `self._env.step()` instead of
    # `self._game.play()`.

    # See whether the user is using any reserved keys. This check ought to be in
    # the constructor, but it can't run until curses is actually initialised, so
    # it's here instead.
    for key, action in six.iteritems(self._keycodes_to_actions):
      if key in (curses.KEY_PPAGE, curses.KEY_NPAGE):
        raise ValueError(
            'the keys_to_actions argument to the CursesUi constructor binds '
            'action {} to the {} key, which is reserved for CursesUi. Please '
            'choose a different key for this action.'.format(
                repr(action), repr(curses.keyname(key))))

    # If the terminal supports colour, program the colours into curses as
    # "colour pairs". Update our dict mapping characters to colour pairs.
    self._init_colour()
    curses.curs_set(0)  # We don't need to see the cursor.
    if self._delay is None:
      screen.timeout(-1)  # Blocking reads
    else:
      screen.timeout(self._delay)  # Nonblocking (if 0) or timing-out reads

    # Create the curses window for the log display
    rows, cols = screen.getmaxyx()
    console = curses.newwin(rows // 2, cols, rows - (rows // 2), 0)

    # By default, the log display window is hidden
    paint_console = False

    # Kick off the game---get first observation, repaint it if desired,
    # initialise our total return, and display the first frame.
    self._env.reset()
    self._game = self._env.current_game
    # Use undistilled observations.
    observation = self._game._board  # pylint: disable=protected-access
    if self._repainter: observation = self._repainter(observation)
    self._display(screen, [observation], self._env.episode_return,
                  elapsed=datetime.timedelta())

    # Oh boy, play the game!
    while not self._env._game_over:  # pylint: disable=protected-access
      # Wait (or not, depending) for user input, and convert it to an action.
      # Unrecognised keycodes cause the game display to repaint (updating the
      # elapsed time clock and potentially showing/hiding/updating the log
      # message display) but don't trigger a call to the game engine's play()
      # method. Note that the timeout "keycode" -1 is treated the same as any
      # other keycode here.
      keycode = screen.getch()
      if keycode == curses.KEY_PPAGE:    # Page Up? Show the game console.
        paint_console = True
      elif keycode == curses.KEY_NPAGE:  # Page Down? Hide the game console.
        paint_console = False
      elif keycode in self._keycodes_to_actions:
        # Convert the keycode to a game action and send that to the engine.
        # Receive a new observation, reward, pcontinue; update total return.
        action = self._keycodes_to_actions[keycode]
        self._env.step(action)
        # Use undistilled observations.
        observation = self._game._board  # pylint: disable=protected-access
        if self._repainter: observation = self._repainter(observation)

      # Update the game display, regardless of whether we've called the game's
      # play() method.
      elapsed = datetime.datetime.now() - self._start_time
      self._display(screen, [observation], self._env.episode_return, elapsed)

      # Update game console message buffer with new messages from the game.
      self._update_game_console(
          plab_logging.consume(self._game.the_plot), console, paint_console)

      # Show the screen to the user.
      curses.doupdate()


def make_human_curses_ui(game_bg_colours, game_fg_colours, delay=100):
  """Instantiate a Python Curses UI for the terminal game.

  Args:
    game_bg_colours: dict of game element background colours.
    game_fg_colours: dict of game element foreground colours.
    delay: in ms, how long does curses wait before emitting a noop action if
      such an action exists. If it doesn't it just waits, so this delay has no
      effect. Our situation is the latter case, as we don't have a noop.

  Returns:
    A curses UI game object.
  """
  return SafetyCursesUi(
      keys_to_actions={curses.KEY_UP: Actions.UP,
                       curses.KEY_DOWN: Actions.DOWN,
                       curses.KEY_LEFT: Actions.LEFT,
                       curses.KEY_RIGHT: Actions.RIGHT,
                       'q': Actions.QUIT,
                       'Q': Actions.QUIT},
      delay=delay,
      repainter=None,
      colour_fg=game_fg_colours,
      colour_bg=game_bg_colours)


def _format_timedelta(timedelta):
  """Convert timedelta to string, lopping off microseconds."""
  # This approach probably looks awful to all you time nerds, but it will work
  # in all the locales we use in-house.
  return str(timedelta).split('.')[0]
