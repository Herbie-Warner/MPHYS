# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:42:10 2024

@author: herbi
"""

import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utilities.Utilities import DELTA


def A(m):
    return m**2 * (-DELTA - 1 +np.log(m**2))