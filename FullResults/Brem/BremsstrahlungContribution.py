# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 23:35:25 2024

@author: herbi
"""
import numpy as np
from Utilities import ME,MGAMMA,MMU,g,stheta
from LOContribution import LO


#YOU HAVE USED ETH = 0.45 ROOT S

def tau(pidotpj,mi,mj):
    alpha = -1/2 * -mi**2/(pidotpj)
    #alpha = -2*pidotpj/(-mj**2)


    return alpha/(alpha*mj**2-mi**2)*np.log(mj**2*alpha**2/mi**2)

def get_omega_cutoff(eth,s):
    val = eth - np.sqrt(s) + np.sqrt(eth**2-MMU**2) -MMU**2/(eth-np.sqrt(s)-np.sqrt(eth**2-MMU**2))
    return -0.5*val

def LIJ_IR(pidotpj,mi,mj):
    tauf = tau(pidotpj,mi,mj)
    pref = -2*pidotpj*tauf
    sval = -2*pidotpj
    
    ethcutoff = 0.45*np.sqrt(sval)
    
    omega = get_omega_cutoff(ethcutoff, sval)
    #print(pref*np.log(2*omega/MGAMMA)**2)
    #print(np.log(2*omega/MGAMMA)**2)
    #print(pref)
    return pref*np.log(2*omega/MGAMMA)**2



def brem_IR_vertex(s,t):
    L12 = LIJ_IR(-s/2,ME,ME)
    L34 = LIJ_IR(-s/2,MMU,MMU)
    pref = g**2/(16*np.pi**2)*stheta**2
    #print(L12,L34)
    contribution = pref*2*(L12+L34)*LO(s,t)
    return contribution
    



sval = 5000000
tval = 250
print(brem_IR_vertex(sval, tval).real)

from VertexCorrections import totalVertex, Dia1E,prefactor,totalVertex

#print(Dia1E(sval, tval)*prefactor)
print(totalVertex(sval, tval).real)



#print(totalVertex(sval, tval).real)
#print(LO(sval,tval))

