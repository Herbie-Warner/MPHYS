# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:15:48 2024

@author: herbi
"""

from Utilities import ME,MGAMMA
import numpy as np

sval = 2000

alpha_plus = -sval/(2*ME**2) * (1+1-2*ME**4/sval**2)
alpha_mins = -sval/(2*ME**2) * (2*ME**4/sval**2)



alpha = alpha_plus

C0IRVAL = -1/2 * alpha/((alpha**2-1)*ME**2) * np.log(alpha**2)*np.log(MGAMMA**2)
print(C0IRVAL)
#print(np.log(MGAMMA**2))

from CFunctions import C0

print(C0(-ME**2, -ME**2, ME, MGAMMA, ME, -sval/2))
