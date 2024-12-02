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

from Lo.LOContribution import LO, LO_GAMMA,LO_Z,LO_inter
from WaveFunctionsRenormalisation.WRContribution import dsigma_WR_E, dsigma_WR_W
from Vertex.VertexCorrections import totalVertexE,totalVertexW
from Vacuum.Vacuum import photon_self_energy,Z_self_energy,mix_self_energy
from Utilities.Utilities import Gev_minus_2_to_mbarns, MW


def templated_plot():
    ECM = MW
    sval = ECM**2
    theta = np.linspace(0, 2 * np.pi, 1000)
    tval = sval / 2 * (np.cos(theta) - 1)
    
    # Replace with your actual functions for these calculations
    LOVALE = LO_GAMMA(sval, tval)*Gev_minus_2_to_mbarns
    LOVALZ = LO_Z(sval, tval)*Gev_minus_2_to_mbarns
    LOVALEZ = LO_inter(sval, tval)*Gev_minus_2_to_mbarns
    
    total = LOVALE+LOVALZ+LOVALEZ
    
    photon_self = photon_self_energy(sval, tval)*Gev_minus_2_to_mbarns
    Z_self = Z_self_energy(sval, tval)*Gev_minus_2_to_mbarns
    mix_self = mix_self_energy(sval, tval)*Gev_minus_2_to_mbarns
    # Convert theta to degrees
    theta *= 180 / np.pi
    
    # Set font
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Create the subdirectory for saving plots
    output_dir = "Vacuum_figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the plot
    figure = plt.figure()
    ax = figure.add_subplot()
    
    #ax.plot(theta, LOVALE, label=r"$d\sigma^0_E$", color='b')
    #ax.plot(theta, LOVALZ, label=r"$d\sigma^0_Z$", color='g')
    #ax.plot(theta, LOVALEZ, label=r"$d\sigma^0_{E Z}$", color='r')
    ax.plot(theta, total, label=r"$d\sigma^0$", color='k')
    
    
       
    ax.plot(theta, photon_self, label=r"$d\sigma^\Delta_E$", color='b')
    ax.plot(theta, Z_self, label=r"$d\sigma^\Delta_Z$", color='g')
    ax.plot(theta, mix_self, label=r"$d\sigma^M$", color='r')
    
    # Update title and labels dynamically
    ax.legend(
    loc='center right',  # Position the legend in the middle-right
    frameon=True,        # Optional: Add a border to the legend
    framealpha=0.7,      # Set the transparency of the legend box (0 = fully transparent, 1 = fully opaque)
    fontsize=10          # Adjust the font size (optional)
)
    ax.set_title(
        rf"Contributions to $e^+ e^- \rightarrow \mu^+ \mu^-$ at LO at $E_{{CM}} = {ECM}$ GeV",
        fontsize=14, weight='bold'
    )
    ax.set_xlabel("Scattering Angle")
    ax.set_ylabel(r"$d\sigma \quad (pb)$")
    
    # Save the plot dynamically
    output_file = os.path.join(output_dir, f"VAC_ECM={ECM}GEV.pdf")
    plt.savefig(output_file, format='pdf')
    
    # Show the plot
    plt.show()
    
    print(f"Plot saved to {output_file}")




templated_plot()

