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

"""Pycolab rendering wrapper for enabling video recording.

This module contains wrappers that allow for simultaneous transformation of
environment observations into agent view (a numpy 2-D array) and human RGB view
(a numpy 3-D array).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pycolab import rendering


class ObservationToArrayWithRGB(object):
  """Convert an `Observation` to a 2-D `board` and 3-D `RGB` numpy array.

  This class is a general utility for converting `Observation`s into 2-D
  `board` representation and 3-D `RGB` numpy arrays. They are returned as a
  dictionary containing the aforementioned keys.
  """

  def __init__(self, value_mapping, colour_mapping):
    """Construct an `ObservationToArrayWithRGB`.

    Builds a callable that will take `Observation`s and emit a dictionary
    containing a 2-D and 3-D numpy array. The rows and columns of the 2-D array
    contain the values obtained after mapping the characters of the original
    `Observation` through `value_mapping`. The rows and columns of the 3-D array
    contain RGB values of the previous 2-D mapping in the [0,1] range.

    Args:
      value_mapping: a dict mapping any characters that might appear in the
          original `Observation`s to a scalar or 1-D vector value. All values
          in this dict must be the same type and dimension. Note that strings
          are considered 1-D vectors, not scalar values.
      colour_mapping: a dict mapping any characters that might appear in the
          original `Observation`s to a 3-tuple of RGB values in the range
          [0,999].

    """
    self._value_mapping = value_mapping
    self._colour_mapping = colour_mapping

    # Rendering functions for the `board` representation and `RGB` values.
    self._renderers = {
        'board': rendering.ObservationToArray(value_mapping=value_mapping,
                                              dtype=np.float32),
        # RGB should be np.uint8, but that will be applied in __call__,
        # since values here are outside of uint8 range.
        'RGB': rendering.ObservationToArray(value_mapping=colour_mapping)
    }

  def __call__(self, observation):
    """Derives `board` and `RGB` arrays from an `Observation`.

    Returns a dict with 2-D `board` and 3-D `RGB` numpy arrays as described in
    the constructor.

    Args:
      observation: an `Observation` from which this method derives numpy arrays.

    Returns:
      a dict containing 'board' and 'RGB' keys as described.

    """
    # Perform observation rendering for agent and for video recording.
    result = {}
    for key, renderer in self._renderers.items():
      result[key] = renderer(observation)

    # Convert to [0, 255] RGB values.
    result['RGB'] = (result['RGB'] / 999.0 * 255.0).astype(np.uint8)
    return result
