# AI safety gridworlds

This is a suite of reinforcement learning environments illustrating various
safety properties of intelligent agents. These environments are
implemented in [pycolab](https://github.com/deepmind/pycolab), a
highly-customisable gridworld game engine with some batteries included.

For more information, see the accompanying [research
paper](https://arxiv.org/pdf/1711.09883.pdf).

## Instructions

1.  Open a new terminal window (`iterm2` on Mac, `gnome-terminal` or `xterm` on
    linux work best, avoid `tmux`/`screen`).
2.  Set the terminal colours to `xterm-256color` by running `export
    TERM=xterm-256color`.
3.  Clone the repository using 
    `git clone --recursive https://github.com/deepmind/ai-safety-gridworlds.git`. 
    This will make sure you get the necessary pycolab submodule as well. 
4.  Choose an environment from the list below and run it by typing 
    `PYTHONPATH=. python -B ai_safety_gridworlds/environments/ENVIRONMENT_NAME.py`.

## Dependencies


* Python 2.7 with enum34 support. We recommend using version 2.7.13.
* Numpy. Our version is 1.13.3.
* [Abseil](https://github.com/abseil/abseil-py) Python common libraries.


## Environments

Our suite includes the following environments.

1.  **Safe interruptibility**: We want to be able to interrupt an agent and
    override its actions at any time. How can we prevent the agent from learning
    to avoid interruptions? `safe_interruptibility.py`
2.  **Avoiding side effects**: How can we incentivize agents to minimize effects
    unrelated to their main objectives, especially those that are irreversible
    or difficult to reverse? `side_effects_sokoban.py`
3.  **Absent supervisor**: How can we ensure that the agent does not behave
    differently depending on whether it is being supervised?
    `absent_supervisor.py`
4.  **Reward gaming**: How can we design agents that are robust to misspecified
    reward functions, for example by modeling their uncertainty about the reward
    function? `boat_race.py` and `tomato_watering.py`
5.  **Self-modification**: Can agents be robust to limited self-modifications,
    for example if they can increase their exploration rate? `whisky-gold.py`
6.  **Distributional shift**: How can we detect and adapt to a data distribution
    that is different from the training distribution? `distributional_shift.py`
7.  **Robustness to adversaries**: How can we ensure the agent's performance
    does not degrade in the presence of adversaries? `friend_foe.py`
8.  **Safe exploration**: How can we ensure satisfying a safety constraint under
    unknown environment dynamics? `island_navigation.py`

Our environments are Markov Decision Processes. All environments use a grid of
size at most 10x10. Each cell in the grid can be empty, or contain a wall or
other objects. These objects are specific to each environment and are explained
in the corresponding section in the paper. The agent is located in one cell on
the grid and in every step the agent takes one of the actions from the action
set A = {left, right, up, down}. Each action modifies the agent's position to
the next cell in the corresponding direction unless that cell is a wall or
another impassable object, in which case the agent stays put.

The agent interacts with the environment in an episodic setting: at the start of
each episode, the environment is reset to its starting configuration (which is
possibly randomized). The agent then interacts with the environment until the
episode ends, which is specific to each environment. We fix the maximal episode
length to 100 steps. Several environments contain a goal cell, depicted as G. If
the agent enters the goal cell, it receives a reward of +50 and the episode
ends. We also provide a default reward of −1 in every time-step to encourage
finishing the episode sooner than later, and use no discounting in the
environment.

In the classical reinforcement learning framework, the agent's objective is to
maximize the cumulative (visible) reward signal. While this is an important part
of the agent's objective, in some problems this does not capture everything that
we care about. Instead of the reward function, we evaluate the agent on the
performance function *that is not observed by the agent*. The performance
function might or might not be identical to the reward function. In real-world
examples, the performance function would only be implicitly defined by the
desired behavior the human designer wishes to achieve, but is inaccessible to
the agent and the human designer.
