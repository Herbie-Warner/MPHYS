# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:02:16 2024

@author: herbi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:08:09 2024

@author: herbi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:04:32 2024

@author: herbi
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Lo.LOTotal import LO_Z_Total, LO_GAMMA_Total, LO_Z_GAMMA_Total
#from Vacuum.VacuumTotal import photon_self_energy_Total, Z_self_energy_Total,mix_self_energy_Total
#from Vertex.VertexCorrectionsTotal import totalVertex
from Utilities.Utilities import Gev_minus_2_to_mbarns

    

def templated_plot():
    domain1 = np.linspace(5, 86.25,1000)
    domain2 = np.linspace(96, 300,1000)
    
    domainFull = np.linspace(5,300,1000)

    s1 = np.square(domain1)
    s2 = np.square(domain2)
    
    LOZ1 = Gev_minus_2_to_mbarns*LO_Z_Total(s1)
    LOZ2 = Gev_minus_2_to_mbarns*LO_Z_Total(s2)
    
    LOg1 = Gev_minus_2_to_mbarns*LO_GAMMA_Total(np.square(domainFull))
 
    
    LOgZ1 = Gev_minus_2_to_mbarns*LO_Z_GAMMA_Total(s1)
    LOgZ2 = Gev_minus_2_to_mbarns*LO_Z_GAMMA_Total(s2)
    
    
    #tot1 = LOZ1+LOg1+LOgZ1
    #tot2 = LOZ2+LOg2+LOgZ2

    plt.rcParams["font.family"] = "Times New Roman"
    output_dir = "LO_figures"
    os.makedirs(output_dir, exist_ok=True)

    figure = plt.figure()
    ax = figure.add_subplot()
    
    #ax.plot(theta, LOVALE, label=r"$d\sigma^0_E$", color='b')
    ax.plot(domain1, LOZ1, label=r"$\sigma^0_Z$", color='g')
    ax.plot(domain2, LOZ2, color='g')
    
    ax.plot(domainFull, LOg1, label=r"$\sigma^0_E$", color='b')

    
    ax.plot(domain1, LOgZ1, label=r"$\sigma^0_{E Z}$", color='r')
    ax.plot(domain2, LOgZ2, color='r')
    

    #ax.plot(theta, LOVALEZ, label=r"$d\sigma^0_{\gamma Z}$", color='r')
   # ax.plot(theta, total, label=r"$d\sigma^0$", color='k')

    ax.legend(
    loc='center right', 
    frameon=True,       
    framealpha=0.7,     
    fontsize=10          
)
    ax.set_title(
        r"Contributions to $\sigma^0$",
        fontsize=14, weight='bold'
    )
    ax.set_xlabel(r"$E_{CM}$ GeV")
    ax.set_ylabel(r"$\sigma \quad (pb)$")

    output_file = os.path.join(output_dir, "LO_Total.pdf")
    plt.savefig(output_file, format='pdf')

    plt.show()  
    print(f"Plot saved to {output_file}")




templated_plot()

