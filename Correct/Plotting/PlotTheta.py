# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:44:50 2024

@author: herbi
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LeadingOrder.LeadingOrderContribution import LeadingOrderTotal,sig_Z,sig_P,sig_ZP
from Core.Quantities import CONVERSION_GEV_TO_mBarns

from VacuumPolarisation.VacuumPolarisationContribution import Z_Self_Energy,P_Self_Energy,ZP_Self_Energy
from Vertex.VertexContribution import VertexCorrectionTotal
#from Box.BoxContribution import boxTotal

from QED.QED import QED

def plotting_LO():
    ECM = 100
    sval = ECM**2
    theta = np.linspace(0, 2 * np.pi, 1000)
    tval = sval / 2 * (np.cos(theta) - 1)
    
    
    
    Z_cont = sig_Z(sval,tval)*CONVERSION_GEV_TO_mBarns
    P_cont = sig_P(sval,tval)*CONVERSION_GEV_TO_mBarns
    ZP_cont = sig_ZP(sval,tval)*CONVERSION_GEV_TO_mBarns
    
    total = LeadingOrderTotal(sval, tval)*CONVERSION_GEV_TO_mBarns
    

    theta *= 180 / np.pi
    

    plt.rcParams["font.family"] = "Times New Roman"
    

    output_dir = "LeadingOrderFigures"
    os.makedirs(output_dir, exist_ok=True)

    figure = plt.figure()
    ax = figure.add_subplot()
    

    ax.plot(theta, total, label=r"$d\sigma^0$", color='k')
    ax.plot(theta, Z_cont, label=r"$d\sigma^0_Z$", color='r')
    ax.plot(theta, P_cont, label=r"$d\sigma^0_\gamma$", color='g')
    ax.plot(theta, ZP_cont, label=r"$d\sigma^0_{\gamma Z}$", color='b')
    
    

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
    
    output_file = os.path.join(output_dir, f"LeadingOrder={ECM}GEV.pdf")
    #plt.savefig(output_file, format='pdf')
    
    # Show the plot
    plt.show()
    
    print(f"Plot saved to {output_file}")

def plotting_Vacuum_Polarisation():
    ECM = 1000
    
    sval = ECM**2
    theta = np.linspace(0, 2 * np.pi, 1000)
    tval = sval / 2 * (np.cos(theta) - 1)
    
    
    
    #Z_cont = sig_Z(sval,tval)*CONVERSION_GEV_TO_mBarns
    #P_cont = sig_P(sval,tval)*CONVERSION_GEV_TO_mBarns
    #ZP_cont = sig_ZP(sval,tval)*CONVERSION_GEV_TO_mBarns
    
    total = LeadingOrderTotal(sval, tval)*CONVERSION_GEV_TO_mBarns
    
    vertex_MPC = VertexCorrectionTotal(sval, tval)*CONVERSION_GEV_TO_mBarns
    vertex = np.array([float(c.real) for c in vertex_MPC])
 
    
    z_self_MPC = Z_Self_Energy(sval, tval)*CONVERSION_GEV_TO_mBarns
    p_self_MPC = P_Self_Energy(sval, tval)*CONVERSION_GEV_TO_mBarns
    zp_self_MPC = ZP_Self_Energy(sval, tval)*CONVERSION_GEV_TO_mBarns
    
    z_self = np.array([float(c.real) for c in z_self_MPC])
    zp_self = np.array([float(c.real) for c in zp_self_MPC])
    p_self = np.array([float(c.real) for c in p_self_MPC])
    
    vac = z_self+zp_self+p_self 

    newtotal = total + vertex + vac
    
    print(np.average(np.divide(newtotal,total)))
    
  

    plt.rcParams["font.family"] = "Times New Roman"
    

    output_dir = "VacuumPolarisationFigures"
    os.makedirs(output_dir, exist_ok=True)

    figure = plt.figure()
    ax = figure.add_subplot()
    theta *= 180 / np.pi
    

    ax.plot(theta, total, label=r"$d\sigma^0$", color='k')
    #ax.plot(theta, Z_cont, label=r"$d\sigma^0_Z$", color='r')
    #ax.plot(theta, P_cont, label=r"$d\sigma^0_\gamma$", color='g')
    #ax.plot(theta, ZP_cont, label=r"$d\sigma^0_{\gamma Z}$", color='b')
    
    ax.plot(theta, vac, label=r"$d\sigma^V$", color='r')
    ax.plot(theta, vertex, label=r"$d\sigma^{Ver}$", color='g')
    #ax.plot(theta, zp_self, label=r"$d\sigma^V_{Z\gamma}$", color='b')
    
    ax.plot(theta, newtotal, label=r"$d\sigma^1$", color='purple')
    

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
    
    output_file = os.path.join(output_dir, f"LeadingOrder={ECM}GEV.pdf")
    #plt.savefig(output_file, format='pdf')
    
    # Show the plot
    plt.show()
    
    #print(f"Plot saved to {output_file}")
    
def plotting_Vertex():
    ECM = 100
    sval = ECM**2
    theta = np.linspace(0, 2 * np.pi, 1000)
    tval = sval / 2 * (np.cos(theta) - 1)
    
    
    
    #Z_cont = sig_Z(sval,tval)*CONVERSION_GEV_TO_mBarns
    #P_cont = sig_P(sval,tval)*CONVERSION_GEV_TO_mBarns
    #ZP_cont = sig_ZP(sval,tval)*CONVERSION_GEV_TO_mBarns
    
    total = LeadingOrderTotal(sval, tval)*CONVERSION_GEV_TO_mBarns  
    z_self_MPC = VertexCorrectionTotal(sval, tval)*CONVERSION_GEV_TO_mBarns
    z_self = [float(c.real) for c in z_self_MPC]

    new_tot = total+z_self

    plt.rcParams["font.family"] = "Times New Roman"
    

    output_dir = "VertexCorrectionFigures"
    os.makedirs(output_dir, exist_ok=True)

    figure = plt.figure()
    ax = figure.add_subplot()
    theta *= 180 / np.pi
    

    ax.plot(theta, total, label=r"$d\sigma^0$", color='k')
    #ax.plot(theta, Z_cont, label=r"$d\sigma^0_Z$", color='r')
    #ax.plot(theta, P_cont, label=r"$d\sigma^0_\gamma$", color='g')
    #ax.plot(theta, ZP_cont, label=r"$d\sigma^0_{\gamma Z}$", color='b')
    
    ax.plot(theta, z_self, label=r"$d\sigma^V$", color='orange')
    
    ax.plot(theta, new_tot, label=r"$d\sigma^1$", color='purple')
    

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
    
    output_file = os.path.join(output_dir, f"LeadingOrder={ECM}GEV.pdf")
    #plt.savefig(output_file, format='pdf')
    
    # Show the plot
    plt.show()
    
    #print(f"Plot saved to {output_file}")


def plotting_box():
    ECM = 10000
    sval = ECM**2
    theta = np.linspace(0.9*np.pi, 2 * np.pi - np.pi/6, 100)
    tval = sval / 2 * (np.cos(theta) - 1)
    
    
    
    #Z_cont = sig_Z(sval,tval)*CONVERSION_GEV_TO_mBarns
    #P_cont = sig_P(sval,tval)*CONVERSION_GEV_TO_mBarns
    #ZP_cont = sig_ZP(sval,tval)*CONVERSION_GEV_TO_mBarns
    
    total = LeadingOrderTotal(sval, tval)*CONVERSION_GEV_TO_mBarns  
    
    z_self_MPC = []
    i = 0
    for t in tval:
        i += 1
        print(i)
        val = boxTotal(sval,t)*CONVERSION_GEV_TO_mBarns
        
        z_self_MPC.append(val)
        print(val)
        print(LeadingOrderTotal(sval, t)*CONVERSION_GEV_TO_mBarns)
        
    
    #z_self_MPC = boxTotal(sval, tval)*CONVERSION_GEV_TO_mBarns
    #z_self = [float(c.real) for c in z_self_MPC]
    z_self = z_self_MPC
    new_tot = total+z_self

    plt.rcParams["font.family"] = "Times New Roman"
    

    output_dir = "VertexCorrectionFigures"
    os.makedirs(output_dir, exist_ok=True)

    figure = plt.figure()
    ax = figure.add_subplot()
    theta *= 180 / np.pi
    

    ax.plot(theta, total, label=r"$d\sigma^0$", color='k')
    #ax.plot(theta, Z_cont, label=r"$d\sigma^0_Z$", color='r')
    #ax.plot(theta, P_cont, label=r"$d\sigma^0_\gamma$", color='g')
    #ax.plot(theta, ZP_cont, label=r"$d\sigma^0_{\gamma Z}$", color='b')
    
    ax.plot(theta, z_self, label=r"$d\sigma^V$", color='orange')
    
    ax.plot(theta, new_tot, label=r"$d\sigma^1$", color='purple')
    

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
    
    output_file = os.path.join(output_dir, f"LeadingOrder={ECM}GEV.pdf")
    #plt.savefig(output_file, format='pdf')
    
    # Show the plot
    plt.show()
    
    #print(f"Plot saved to {output_file}")
    
    
def plotting_QED():
    ECM = 1000
    sval = ECM**2
    theta = np.linspace(0, 2 * np.pi, 1000)
    tval = sval / 2 * (np.cos(theta) - 1)
    
    
    
    total = QED(sval, tval)*CONVERSION_GEV_TO_mBarns
    

    theta *= 180 / np.pi
    

    plt.rcParams["font.family"] = "Times New Roman"
    

    output_dir = "LeadingOrderFigures"
    os.makedirs(output_dir, exist_ok=True)

    figure = plt.figure()
    ax = figure.add_subplot()
    

    ax.plot(theta, total, label=r"$d\sigma^0$", color='k')
    #ax.plot(theta, Z_cont, label=r"$d\sigma^0_Z$", color='r')
    #ax.plot(theta, P_cont, label=r"$d\sigma^0_\gamma$", color='g')
    #ax.plot(theta, ZP_cont, label=r"$d\sigma^0_{\gamma Z}$", color='b')
    
    

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
    
    output_file = os.path.join(output_dir, f"LeadingOrder={ECM}GEV.pdf")
    #plt.savefig(output_file, format='pdf')
    
    # Show the plot
    plt.show()
    
    print(f"Plot saved to {output_file}")
    
    
from mpl_toolkits.mplot3d import Axes3D 

def thd_graph():
    num = 10000
    x = np.linspace(15,150,num)
    y = np.linspace(0,2*np.pi,num)
    X, Y = np.meshgrid(x, y)
    
    sval = np.square(X)
    tval = sval/2 * (np.cos(Y)-1)
    
    total = LeadingOrderTotal(sval,tval).real*CONVERSION_GEV_TO_mBarns
    
    #vertex_MPC = VertexCorrectionTotal(sval, tval)*CONVERSION_GEV_TO_mBarns
    #vertex = np.array([float(c.real) for c in vertex_MPC])
    
    #
    
    Z = total
    Y *= 180/np.pi
    
    # Create a figure for the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    plt.rcParams["font.family"] = "Times New Roman"
    ax.plot_surface(X, Y, Z, cmap='viridis')
    
    
    
    # Add labels
    ax.set_xlabel(r'$E_{CM}$ GeV')
    ax.set_ylabel(r'$\theta$')
    ax.set_zlabel(r'$d\sigma \quad (pb)$')
    
   
    ax.set_title(
        rf"Leading Order Contribution to $e^+ e^- \rightarrow \mu^+ \mu^-$",
        fontsize=14, weight='bold'
    )
    
    # Show the plot
    plt.savefig("LO_3D.pdf", format='pdf')
    plt.show()


#thd_graph()
plotting_Vacuum_Polarisation()
#plotting_Vertex()
#plotting_LO()
#plotting_box()
#plotting_QED()