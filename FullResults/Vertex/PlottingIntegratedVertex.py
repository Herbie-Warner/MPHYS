# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:44:46 2024

@author: herbi
"""



import numpy as np
import matplotlib.pyplot as plt
from LOContribution import LO
from VertexCorrectionsTotal import totalVertexE,totalVertexW
from Utilities import stheta,ctheta,vtheta, MZ


def gamma(ECM):
    return stheta**4 / (12*np.pi*ECM**2)

def Z(ECM):
    return ECM**2 * (1+vtheta**2)**2 / (3072*np.pi*ctheta**4*(ECM**2-MZ**2)**2)

def ZGamma(ECM):
    return stheta**2 * vtheta**2 / (96*np.pi*ctheta**2 * (ECM**2-MZ**2))


def plot():
    ECM = 1000
    sval = ECM**2
    
    theta = np.linspace(0,2*np.pi,1000)
    tval = sval/2 * (np.cos(theta)-1)
    
    
    LOVAL = LO(sval,tval)
    VVal = totalVertexE(sval,tval).real
    WVal = totalVertexW(sval, tval).real
    tot = VVal + WVal
    
    
    
   
  
    figure = plt.figure()
    ax = figure.add_subplot()
    
    ax.plot(theta,LOVAL,label="LO")
    ax.plot(theta,VVal,label="VVAL")
    ax.plot(theta,WVal,label="WVAL")
    print(np.average(LOVAL))
    print(np.average(VVal))
    ax.plot(theta,tot,label="tot")

    ax.legend()
    plt.show()
    
  
#
"""
ECM = 100
sval = ECM**2

tval = sval/2 * (np.cos(np.pi)-1)
#print(tval)   
VVal = totalVertexE(sval,tval)
#WVal = totalVertexW(sval, tval)
#tot = VVal + WVal
#print(tot)
#
"""


def templated_plot():
    ECM = np.linspace(10,88,50)
    ECM2 = np.linspace(93,200,50)
    sval2 = ECM2**2
    #ECM = np.linspace(10,90,50)
    sval = ECM**2
    
  
    #10
    #100
    #13600
    #LOVAL = LO(sval,tval)
    
    VVAL = []
    WVAL = []
    
    
    VVAL2 = []
    WVAL2 = []
    
    
    i = 0
    for s in sval:
        print(i)
        i +=1 
        VVAL.append(-(totalVertexE(s,0).real))
        WVAL.append(-(totalVertexW(s,0).real))
        
        
    for s in sval2:
        VVAL2.append(-(totalVertexE(s,0).real))
        WVAL2.append(-(totalVertexW(s,0).real))
       
    
  

  
    
    LOVAL = gamma(ECM)+Z(ECM)+ZGamma(ECM)
    
    LOVAL2 = gamma(ECM2)+Z(ECM2)+ZGamma(ECM2)
    
    total = np.add(np.array(VVAL),np.array(WVAL))
    
    total2 = np.add(np.array(VVAL2),np.array(WVAL2))

    plt.rcParams["font.family"] = "Times New Roman"

    # Create the figure and axis
    figure = plt.figure()
    ax = figure.add_subplot()

    # Plot the data with customized labels in LaTeX
    
    ax.plot(ECM, VVAL, label=r"$V_e$",color='g')      # Gamma in LaTeX
    ax.plot(ECM, WVAL, label=r"$V_z$",color='r')
    ax.plot(ECM,total,label="V",color='k')
    #ax.plot(ECM,LOVAL,label="LO",color='y')
    
    
    ax.plot(ECM2, VVAL2,color='g')      # Gamma in LaTeX
    ax.plot(ECM2, WVAL2, color='r')
    ax.plot(ECM2,total2,color='k')
    #ax.plot(ECM2,LOVAL2,color='y')
    
    # Add legend and title with LaTeX formatting
    ax.legend()
    ax.set_title(r"Contributions to $e^+ e^- \rightarrow \mu^+ \mu^-$ at LO at $E_{CM} = 13.6$ TeV", fontsize=14, weight='bold')
    ax.set_xlabel("Scattering Angle")
    # Show the plot
    #plt.savefig("ECM=13.6TEV.pdf",format='pdf')
    plt.show()




#templated_plot()

GLOBAL_HEIGHT = 1e-7

def reduce_height0(X,Y):
    X_min, X_max = 1, 90
    Y_max = GLOBAL_HEIGHT
    
    # Apply domain and height filters
    domain_mask = (X > X_min) & (X < X_max)
    height_mask = Y < Y_max
    combined_mask = domain_mask & height_mask
    
    X_filtered = X[combined_mask]
    Y_filtered = Y[combined_mask]

    return X_filtered, Y_filtered


def reduce_height1(X,Y):
    X_min, X_max = 92, 55000
    Y_max = GLOBAL_HEIGHT
    
    # Apply domain and height filters
    domain_mask = (X > X_min) & (X < X_max)
    height_mask = Y < Y_max
    combined_mask = domain_mask & height_mask
    
    X_filtered = X[combined_mask]
    Y_filtered = Y[combined_mask]

    return X_filtered, Y_filtered

def new_plot():
     plt.rcParams["font.family"] = "Times New Roman"

     figure = plt.figure()
     ax = figure.add_subplot()
     
     number = 4000
     
     max_end = 400
    
     domain_LO1 = np.linspace(2.45, 83.5,number)
     domain_LO2 = np.linspace(100, max_end,number)
     
     val_LO1 = LO(np.square(domain_LO1),0)
     val_LO2 = LO(np.square(domain_LO2),0)
     
     X,Y = reduce_height0(domain_LO1, val_LO1)
     ax.plot(X,Y,label=r"$\sigma_{LO}$",color='g')
     
     X,Y = reduce_height1(domain_LO2, val_LO2)
     ax.plot(X,Y,color='g')
     
     
     domain_VE1 = np.linspace(20, 90.999999,number)
     domain_VE2 = np.linspace(91.0000001, max_end,number)
     
     val_VE1 = totalVertexE(np.square(domain_VE1),0)
     val_VE2 = totalVertexE(np.square(domain_VE2),0)
     
     X,Y = reduce_height0(domain_VE1, val_VE1)
     ax.plot(X,-Y,label=r"$\sigma^\nu_E$",color='b')
     
     X,Y = reduce_height1(domain_VE2, val_VE2)
     ax.plot(X,-Y,color='b')
     
     
     domain_VW1 = np.linspace(15, 90.999999,number)
     domain_VW2 = np.linspace(91.0000001, max_end,number)
     
     val_VW1 = -totalVertexW(np.square(domain_VW1),0)
     val_VW2 = -totalVertexW(np.square(domain_VW2),0)
     
     X,Y = reduce_height0(domain_VW1, val_VW1)
  
     ax.plot(X,Y,label=r"$\sigma^\nu_W$",color='r')
     
     X,Y = reduce_height1(domain_VW2, val_VW2)
     ax.plot(X,Y,color='r')
     
     total_domain = np.linspace(20,83.5,number)
     
     totval = LO(np.square(total_domain),0) - totalVertexE(np.square(total_domain), 0) -  totalVertexW(np.square(total_domain), 0)
     
     X,Y = reduce_height0(total_domain, totval)
     ax.plot(X,Y,color='orange',label=r"$\sigma_{Tot}$")
     
     
     total_domain = np.linspace(100,max_end,number)
     
     totval = LO(np.square(total_domain),0) - totalVertexE(np.square(total_domain), 0) -  totalVertexW(np.square(total_domain), 0)
     
     X,Y = reduce_height1(total_domain, totval)
     ax.plot(X,Y,color='orange')
     
     
    
     
     
     
     #ax.plot(ECM, VVAL, label=r"$V_e$",color='g')      # Gamma in LaTeX
     #ax.plot(ECM, WVAL, label=r"$V_z$",color='r')
     #ax.plot(ECM,total,label="V",color='k')
     #ax.plot(ECM,LOVAL,label="LO",color='y')
     
     
     #ax.plot(ECM2, VVAL2,color='g')      # Gamma in LaTeX
     #ax.plot(ECM2, WVAL2, color='r')
     #ax.plot(ECM2,total2,color='k')
     #ax.plot(ECM2,LOVAL2,color='y')
     
     # Add legend and title with LaTeX formatting
     ax.legend()
     ax.set_title(r"Vertex Contribution and LO", fontsize=14, weight='bold')
     ax.set_xlabel(r"$E_{CM} GeV$")
     # Show the plot
     #plt.savefig("ECM=13.6TEV.pdf",format='pdf')
     plt.savefig("Total.pdf",format='pdf')
     plt.show()


new_plot()
