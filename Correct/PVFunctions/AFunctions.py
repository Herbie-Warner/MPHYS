# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:32:54 2024

@author: herbi
"""

import gmpy2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Quantities import DELTA
from Core.Precision import HPC


def A(m):
    mass = HPC(m)
    return mass**2 * (-DELTA - 1 +gmpy2.log(mass**2))