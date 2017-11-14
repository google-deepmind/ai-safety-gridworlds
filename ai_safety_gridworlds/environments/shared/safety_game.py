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

"""Helpers for creating safety environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import enum
import numpy as np

from ai_safety_gridworlds.environments.shared import observation_distiller
from ai_safety_gridworlds.environments.shared.rl import array_spec as specs
from ai_safety_gridworlds.environments.shared.rl import pycolab_interface
from ai_safety_gridworlds.environments.shared.termination_reason_enum import TerminationReason
from pycolab import ascii_art
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites


class Actions(enum.IntEnum):
  """Enum for actions all the players can take."""
  UP = 0
  DOWN = 1
  LEFT = 2
  RIGHT = 3
  # Human only.
  QUIT = 4


# Colours common in all environments.
GAME_BG_COLOURS = {' ': (858, 858, 858),  # Environment floor.
                   '#': (599, 599, 599),  # Environment walls.
                   'A': (0, 706, 999),    # Player character.
                   'G': (0, 823, 196)}    # Goal.
GAME_FG_COLOURS = {' ': (858, 858, 858),
                   '#': (599, 599, 599),
                   'A': (0, 0, 0),
                   'G': (0, 0, 0)}

# The maximal value of an action that an agent can take.
AGENT_LAST_ACTION = Actions.RIGHT.value

# Some constants to use with the environment_data dictionary to avoid
ENV_DATA = 'environment_data'
ACTUAL_ACTIONS = 'actual_actions'
CURSES = 'curses'
TERMINATION_REASON = 'termination_reason'
HIDDEN_REWARD = 'hidden_reward'

# Constants for the observations dictionary to the agent.
EXTRA_OBSERVATIONS = 'extra_observations'


class SafetyEnvironment(pycolab_interface.Environment):
  """Base class for safety gridworld environments.

  Environments implementing this base class initialize the Python environment
  API v2 and serve as a layer in which we can put various modifications of
  pycolab inputs and outputs, such as *additional information* passed
  from/to the environment that does not fit in the traditional observation
  channel. It also allows for overwriting of default methods such as step() and
  reset().

  Each new environment must implement a subclass of this class, and at the very
  least call the __init__ method of this class with corresponding parameters, to
  instantiate the python environment API around the pycolab game.
  """

  def __init__(self,
               game_factory,
               game_bg_colours,
               game_fg_colours,
               actions=None,
               value_mapping=None,
               environment_data=None,
               repainter=None,
               max_iterations=100):
    """Initialize a Python v2 environment for a pycolab game factory.

    Args:
      game_factory: a function that returns a new pycolab `Engine`
        instance corresponding to the game being played.
      game_bg_colours: a dict mapping game characters to background RGB colours.
      game_fg_colours: a dict mapping game characters to foreground RGB colours.
      actions: a tuple of ints, indicating an inclusive range of actions the
        agent can take. Defaults to (0, AGENT_LAST_ACTION).
      value_mapping: a dictionary mapping characters from the game ascii map
        into floats. Used to control how the agent sees the game ascii map, e.g.
        if we are not making a difference between environment background and
        walls in terms of values the agent sees for those blocks, we can map
        them to the same value. Defaults to mapping characters to their ascii
        codes.
      environment_data: dictionary of data that is passed to the pycolab
        environment implementation and is used as a shared object that allows
        each wrapper to communicate with their environment. This object can hold
        additional information about the state of the environment that can even
        persists through episodes, but some particular keys are erased at each
        new episode.
      repainter: a callable that converts `rendering.Observation`s to different
        `rendering.Observation`s, or None if no such conversion is required.
        This facility is normally used to change the characters used to depict
        certain game elements, and a `rendering.ObservationCharacterRepainter`
        object is a convenient way to accomplish this conversion. For more
        information, see pycolab's `rendering.py`.
      max_iterations: the maximum number of steps for one episode.
    """
    if environment_data is None:
      self._environment_data = {}
    else:
      self._environment_data = environment_data
    # Used to store agent performance per episode. Note that agent performance
    # metric might not be equal to the reward obtained.
    self._episodic_performances = []
    # Total environment reward for the current episode.
    self._episode_return = 0
    # Keys to clear from environment_data at start of each episode.
    self._keys_to_clear = [TERMINATION_REASON, ACTUAL_ACTIONS]

    if actions is None:
      actions = (0, AGENT_LAST_ACTION)

    if value_mapping is None:
      value_mapping = {chr(i): i for i in range(256)}
    self._value_mapping = value_mapping

    array_converter = observation_distiller.ObservationToArrayWithRGB(
        value_mapping=value_mapping,
        colour_mapping=game_bg_colours)

    super(SafetyEnvironment, self).__init__(
        game_factory=game_factory,
        discrete_actions=actions,
        default_reward=0,
        observation_distiller=pycolab_interface.Distiller(
            repainter=repainter,
            array_converter=array_converter),
        max_iterations=max_iterations)

  @property
  def environment_data(self):
    return self._environment_data

  @property
  def current_game(self):
    return self._current_game

  @property
  def episode_return(self):
    return self._episode_return

  def _compute_observation_spec(self):
    """Helper for `__init__`: compute our environment's action spec."""
    # This method needs to be overwritten because the parent's method checks
    # all the items in the observation and chokes on the `environment_data`.

    # Start an environment, examine the values it gives to us, and reset things
    # back to default.
    timestep = self.reset()
    observation_spec = {k: specs.ArraySpec(v.shape, v.dtype, name=k)
                        for k, v in timestep.observation.iteritems()
                        if k != EXTRA_OBSERVATIONS}
    observation_spec[EXTRA_OBSERVATIONS] = dict()
    self._drop_last_episode()
    return observation_spec

  def get_overall_performance(self, default=None):
    """Returns the performance measure of the agent across all episodes.

    The agent performance metric might not be equal to the reward obtained,
    depending if the environment has a hidden reward function or not.

    Args:
      default: value to return if performance is not yet calculated (i.e. None).

    Returns:
      A float if performance is calculated, None otherwise (if no default).
    """
    if len(self._episodic_performances) < 1:
      return default
    return float(self._calculate_overall_performance())

  def get_last_performance(self, default=None):
    """Returns the last measured performance of the agent.

    The agent performance metric might not be equal to the reward obtained,
    depending if the environment has a hidden reward function or not.

    This method will return the last calculated performance metric.
    When this metric was calculated will depend on 2 things:
      * Last time the timestep step_type was LAST (so if the episode is not
          finished, the metric will be for one of the previous episodes).
      * Whether the environment calculates the metric for every episode, or only
          does it for some (for example, in safe interruptibility, the metric is
          only calculated on episodes where the agent was not interrupted).

    Args:
      default: value to return if performance is not yet calculated (i.e. None).

    Returns:
      A float if performance is calculated, None otherwise (if no default).
    """
    if len(self._episodic_performances) < 1:
      return default
    return float(self._episodic_performances[-1])

  def _calculate_overall_performance(self):
    """Calculates the agent performance across all the episodes.

    By default, the method will return the average across all episodes.
    You should override this method if you want to implement some other way of
    calculating the overall performance.

    Returns:
      A float value summarizing the performance of the agent.
    """
    return sum(self._episodic_performances) / len(self._episodic_performances)

  def _calculate_episode_performance(self, timestep):
    """Calculate performance metric for the agent for the current episode.

    Default performance metric is the average episode reward. You should
    override this method and implement your own if it differs from the default.

    Args:
      timestep: instance of environment.TimeStep
    """
    self._episodic_performances.append(self._episode_return)

  def _get_hidden_reward(self, default_reward=0):
    """Extract the hidden reward from the plot of the current episode."""
    return self.current_game.the_plot.get(HIDDEN_REWARD, default_reward)

  def _clear_hidden_reward(self):
    """Delete hidden reward from the plot."""
    self.current_game.the_plot.pop(HIDDEN_REWARD, None)

  def _process_timestep(self, timestep):
    """Do timestep preprocessing before sending it to the agent.

    This method stores the cumulative return and makes sure that the
    `environment_data` is included in the observation.

    If you are overriding this method, make sure to call `super()` to include
    this code.

    Args:
      timestep: instance of environment.TimeStep

    Returns:
      Preprocessed timestep.
    """
    # Reset the cumulative episode reward.
    if timestep.first():
      self._episode_return = 0
      self._clear_hidden_reward()
      # Clear the keys in environment data from the previous episode.
      for key in self._keys_to_clear:
        self._environment_data.pop(key, None)
    # Add the timestep reward for internal wrapper calculations.
    if timestep.reward:
      self._episode_return += timestep.reward
    extra_observations = self._get_agent_extra_observations()
    if ACTUAL_ACTIONS in self._environment_data:
      extra_observations[ACTUAL_ACTIONS] = (
          self._environment_data[ACTUAL_ACTIONS])
    if timestep.last():
      # Include the termination reason for the episode if missing.
      if TERMINATION_REASON not in self._environment_data:
        self._environment_data[TERMINATION_REASON] = TerminationReason.MAX_STEPS
      extra_observations[TERMINATION_REASON] = (
          self._environment_data[TERMINATION_REASON])
    timestep.observation[EXTRA_OBSERVATIONS] = extra_observations
    # Calculate performance metric if the episode has finished.
    if timestep.last():
      self._calculate_episode_performance(timestep)
    return timestep

  def _get_agent_extra_observations(self):
    """Overwrite this method to give additional information to the agent."""
    return {}

  def reset(self):
    timestep = super(SafetyEnvironment, self).reset()
    return self._process_timestep(timestep)

  def step(self, actions):
    timestep = super(SafetyEnvironment, self).step(actions)
    return self._process_timestep(timestep)


class SafetyBackdrop(plab_things.Backdrop):
  """The backdrop for the game.

  Clear some values in the_plot.
  """

  def update(self, actions, board, layers, things, the_plot):
    super(SafetyBackdrop, self).update(actions, board, layers, things, the_plot)
    PolicyWrapperDrape.plot_clear_actions(the_plot)


class SafetySprite(prefab_sprites.MazeWalker):
  """A generic `Sprite` for objects that move in safety environments.

  Sprites in safety environments that can move, but do not represent the agent,
  should inherit from this class. Sprites that represent the agent should
  inherit from AgentSafetySprite class.

  This `Sprite` has logic tying actions to `MazeWalker` motion action helper
  methods, which keep the sprites from walking on top of obstacles.

  Its main purpose is to wrap the MazeWalker and get hold of the
  environment_data and original_board variables.
  """

  def __init__(self, corner, position, character,
               environment_data, original_board,
               impassable='#'):
    """Initialize SafetySprite.

    Args:
      corner: same as in pycolab sprite.
      position: same as in pycolab sprite.
      character: same as in pycolab sprite.
      environment_data: dictionary of data that is passed to the pycolab
        environment and is used as a shared object that allows each wrapper to
        communicate with their environment.
      original_board: original ascii representation of the board, to avoid using
        layers for checking position of static elements on the board.
      impassable: the character that the agent can't traverse.
    """
    super(SafetySprite, self).__init__(
        corner, position, character, impassable=impassable)
    self._environment_data = environment_data
    self._original_board = original_board

  @abc.abstractmethod
  def update(self, actions, board, layers, backdrop, things, the_plot):
    """See pycolab Sprite class documentation."""
    pass


class AgentSafetySprite(SafetySprite):
  """A generic `Sprite` for agents in safety environments.

  Main purpose is to define some generic behaviour around agent sprite movement,
  action handling and reward calculation.
  """

  def __init__(self, corner, position, character,
               environment_data, original_board,
               impassable='#'):
    """Initialize AgentSafetySprite.

    Args:
      corner: same as in pycolab sprite.
      position: same as in pycolab sprite.
      character: same as in pycolab sprite.
      environment_data: dictionary of data that is passed to the pycolab
        environment and is used as a shared object that allows each wrapper to
        communicate with their environment.
      original_board: original ascii representation of the board, to avoid using
        layers for checking position of static elements on the board.
      impassable: the character that the agent can't traverse.
    """
    super(AgentSafetySprite, self).__init__(
        corner, position, character, environment_data, original_board,
        impassable=impassable)
    self._environment_data = environment_data
    self._original_board = original_board

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop  # Unused.

    if actions is None:
      return

    if actions == Actions.QUIT:
      self._environment_data[TERMINATION_REASON] = TerminationReason.QUIT
      the_plot.terminate_episode()
      return

    # Start by collecting the action chosen by the agent.
    # First look for an entry ACTUAL_ACTIONS in the the_plot dictionary.
    # If none, then use the provided actions instead.
    agent_action = PolicyWrapperDrape.plot_get_actions(the_plot, actions)

    # Remember the actual action so as to notify the agent so that it can
    # update on the action that was actually taken.
    self._environment_data[ACTUAL_ACTIONS] = agent_action

    # Perform the actual action in the environment
    # Comparison between an integer and Actions is allowed because Actions is
    # an IntEnum
    if agent_action == Actions.UP:       # go upward?
      self._north(board, the_plot)
    elif agent_action == Actions.DOWN:   # go downward?
      self._south(board, the_plot)
    elif agent_action == Actions.LEFT:   # go leftward?
      self._west(board, the_plot)
    elif agent_action == Actions.RIGHT:  # go rightward?
      self._east(board, the_plot)

    self.update_reward(actions, agent_action, layers, things, the_plot)

  def update_reward(self, proposed_actions, actual_actions,
                    layers, things, the_plot):
    """Updates the reward after the actions have been processed.

    Children should most likely define this method.

    Args:
      proposed_actions: actions that were proposed by the agent.
      actual_actions: action that is actually carried out in the environment.
        The two are likely to be the same unless a PolicyWrapperDrape changes
        the proposed actions.
      layers: as elsewhere.
      things: as elsewhere.
      the_plot: as elsewhere.
    """
    pass


class EnvironmentDataSprite(plab_things.Sprite):
  """A generic `Sprite` class for safety environments.

  All stationary Sprites in the safety environments should derive from this
  class.

  Its only purpose is to get hold of the environment_data dictionary and
  original_board variables.
  """

  def __init__(self, corner, position, character,
               environment_data, original_board):
    """Initialize environment data sprite.

    Args:
      corner: same as in pycolab sprite.
      position: same as in pycolab sprite.
      character: same as in pycolab sprite.
      environment_data: dictionary of data that is passed to the pycolab
        environment and is used as a shared object that allows each wrapper to
        communicate with their environment.
      original_board: original ascii representation of the board, to avoid using
        layers for checking position of static elements on the board.
    """
    super(EnvironmentDataSprite, self).__init__(corner, position, character)
    self._original_board = original_board
    self._environment_data = environment_data

  def update(self, actions, board, layers, backdrop, things, the_plot):
    """See parent class documentation."""
    pass


class EnvironmentDataDrape(plab_things.Drape):
  """A generic `Drape` class for safety environments.

  All Drapes in the safety environments should derive from this class.

  Its only purpose is to get hold of the environment_data and
  original_board variables.
  """

  def __init__(self, curtain, character,
               environment_data, original_board):
    """Initialize environment data drape.

    Args:
      curtain: same as in pycolab drape.
      character: same as in pycolab drape.
      environment_data: dictionary of data that is passed to the pycolab
        environment and is used as a shared object that allows each wrapper to
        communicate with their environment.
      original_board: original ascii representation of the board, to avoid using
        layers for checking position of static elements on the board.
    """
    super(EnvironmentDataDrape, self).__init__(curtain, character)
    self._original_board = original_board
    self._environment_data = environment_data

  def update(self, actions, board, layers, backdrop, things, the_plot):
    """See parent class documentation."""
    pass


class PolicyWrapperDrape(EnvironmentDataDrape):
  """A `Drape` parent class for policy wrappers.

  Policy wrappers change the entry ACTUAL_ACTIONS in the the_plot
  dictionary.
  Calls the child method `get_actual_action` with the current action
  (which may already have been modified by another sprite)
  and update the current value in the dictionary.
  This value may be used by the agent sprite in place of the agent's action.
  """

  __metaclass__ = abc.ABCMeta

  ACTIONS_KEY = ACTUAL_ACTIONS

  def __init__(self, curtain, character,
               environment_data, original_board, agent_character):
    """Initialize policy wrapper drape.

    Args:
      curtain: same as in pycolab drape.
      character: same as in pycolab drape.
      environment_data: dictionary of data that is passed to the pycolab
        environment and is used as a shared object that allows each wrapper to
        communicate with their environment.
      original_board: original ascii representation of the board, to avoid using
        layers for checking position of static elements on the board.
      agent_character: the ascii character for the agent.
    """
    super(PolicyWrapperDrape, self).__init__(
        curtain, character, environment_data, original_board)
    self._agent_character = agent_character

  def update(self, actions, board, layers, backdrop, things, the_plot):
    agent_action = self.plot_get_actions(the_plot, actions)

    if self._agent_character is not None:
      pos = things[self._agent_character].position
      # If the drape applies globally to all tiles instead of a specific tile,
      # redefine this function without the if statement on the following line.
      # (See example in 'whisky_gold.py.)
      if self.curtain[pos]:
        the_plot[self.ACTIONS_KEY] = self.get_actual_actions(
            agent_action, things, the_plot)

  @abc.abstractmethod
  def get_actual_actions(self, actions, things, the_plot):
    """Takes the actions and returns new actions.

    A child `PolicyWrapperDrape` must implement this method.
    The PolicyWrapperDrapes are chained and can all change these actions.
    The actual actions returned by one drape are the actions input to the next
    one.

    See contrarian.py for a usage example.

    Args:
      actions: either the actions output by the agent (if no drape have modified
        them), or the actions modified by a drape (policy wrapper).
      things: Sprites, Drapes, etc.
      the_plot: the Plot, as elsewhere.
    """
    pass

  @classmethod
  def plot_get_actions(cls, the_plot, actions):
    return the_plot.get(cls.ACTIONS_KEY, actions)

  @classmethod
  def plot_set_actions(cls, the_plot, actions):
    the_plot[cls.ACTIONS_KEY] = actions

  @classmethod
  def plot_clear_actions(cls, the_plot):
    if cls.ACTIONS_KEY in the_plot:
      del the_plot[cls.ACTIONS_KEY]


# Helper function used in various files
def timestep_termination_reason(timestep, default=None):
  return timestep.observation[EXTRA_OBSERVATIONS].get(
      TERMINATION_REASON, default)


def add_hidden_reward(the_plot, reward, default=0):
  """Adds a hidden reward, analogous to pycolab add_reward.

  Args:
     the_plot: the game Plot object.
     reward: numeric value of the hidden reward.
     default: value with which to initialize the hidden reward variable.
  """
  the_plot[HIDDEN_REWARD] = the_plot.get(HIDDEN_REWARD, default) + reward


def terminate_episode(the_plot, environment_data,
                      reason=TerminationReason.TERMINATED, discount=0.0):
  """Tells the pycolab game engine to terminate the current episode.

  Args:
    the_plot: the game Plot object.
    environment_data: dict used to pass around data in a single episode.
    reason: termination reason for the episode.
    discount: discount for the last observation.
  """
  environment_data[TERMINATION_REASON] = reason
  the_plot.terminate_episode(discount=discount)


def make_safety_game(
    environment_data,
    the_ascii_art,
    what_lies_beneath,
    backdrop=SafetyBackdrop,
    sprites=None,
    drapes=None,
    update_schedule=None,
    z_order=None):
  # Keep a still copy of the initial board as a numpy array
  original_board = np.array(map(list, the_ascii_art[:]))
  return ascii_art.ascii_art_to_game(
      the_ascii_art,
      what_lies_beneath,
      sprites=None if sprites is None
      else {k: ascii_art.Partial(args[0],
                                 environment_data,
                                 original_board,
                                 *args[1:])
            for k, args in sprites.items()},
      drapes=None if drapes is None
      else {k: ascii_art.Partial(args[0],
                                 environment_data,
                                 original_board,
                                 *args[1:])
            for k, args in drapes.items()},
      backdrop=backdrop,
      update_schedule=update_schedule,
      z_order=z_order)
