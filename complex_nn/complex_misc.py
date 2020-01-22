#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Support functions for the polynomial cancellation
"""

import numpy as np
import math


def calc_odd_powers(min_power, max_power):
    """Compute how many odd powers we have in between a min exponent and max
    exponent to determine the number of weights in the Hammerstein type
    neural network.
    """

    n_powers = int((max_power + 1) / 2) - int((min_power + 1) / 2) + 1

    return n_powers
