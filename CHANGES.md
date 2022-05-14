# ai\_safety\_gridworlds changelog

## Version 2.0 - Saturday 14. May 2022

* Refactored code for more consistency across environments. 
* Added the following flags to more environments: level, max_iterations, noops. 
* Added safety_ui_ex.make_human_curses_ui_with_noop_keys() method which enables human player to perform no-ops using keyboard. 
* Added SafetyCursesUiEx class which enables printing various custom drape and sprite metrics on the screen. 
* Started extending the maps and implementing multi-objective goals for various environments. 
* In particular, island_navigation_ex.py has been implemented. The latter has now food and drink sources with satiation and deficit aspects in the agent, as well as sustainability aspect in the environment. All these aspects can be turned on and off, as well as their parameters can be configured using flags. Additionally the environment has gold and silver sources.

## Version 1.5 - Tuesday, 13. October 2020

* Corrections for the side_effects_sokoban wall penalty calculation.
* Added new variants for the conveyor_belt and side_effects_sokoban environments.

## Version 1.4 - Tuesday, 13. August 2019

* Added the rocks_diamonds environment.

## Version 1.3.1 - Friday, 12. July 2019

* Removed movement reward in conveyor belt environments.
* Added adjustment of the hidden reward for sushi_goal at the end of the episode to make the performance scale consistent with other environments.
* Added tests for the sushi_goal variant.

## Version 1.3 - Tuesday, 30. April 2019

* Added a new variant of the conveyor_belt environment - *sushi goal*.
* Added optional NOOPs in conveyor_belt and side_effects_sokoban environments.


## Version 1.2 - Wednesday, 22. August 2018

* Python3 support!
* Compatibility with the newest version of pycolab.

Please make sure to see the new installation instructions in [README.md](https://github.com/deepmind/ai-safety-gridworlds/blob/master/README.md) in order to update to the correct version of pycolab.

## Version 1.1 - Monday, 25. June 2018

* Added a new side effects environment - **conveyor_belt.py**, described in
  the accompanying paper: [Measuring and avoiding side effects using relative reachability](https://arxiv.org/abs/1806.01186).

