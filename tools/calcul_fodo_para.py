# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:57:06 2024

@author: LS276867
"""

#Lattices for Collider Storage Ring
#Separated-function FODO cell 

#f: focal length of quadrupoles
#Lp: length of the cell, quads separated by Lp/2
#mu: phase advance of the cell

import numpy as np

mu=np.pi/2
Lp=24.4783
theta=0.0302
f= Lp/(4*np.sin(mu/2))
beta_max=Lp*(1+np.sin(mu/2))/np.sin(mu)
beta_min=Lp*(1-np.sin(mu/2))/np.sin(mu)
alpha_max= (-1-np.sin(mu/2))/np.cos(mu/2)
alpha_min= (-1-np.sin(mu/2))/np.cos(mu/2)
D_max=Lp*theta*(1+np.sin(mu/2)/2)/(4*(np.sin(mu/2))**2)
D_min=Lp*theta*(1-np.sin(mu/2)/2)/(4*(np.sin(mu/2))**2)
D_av= Lp*theta/4*( 1/(np.sin(mu/2))**2 - 1/12)
mcf=theta*D_av/Lp
chromaticity=-1/np.pi*np.tan(mu/2)
print('f=',f)
print('beta_max=',beta_max)
print('beta_min=',beta_min)
print('alpha_max=',alpha_max)
print('alpha_min=',alpha_min)
print('D_max=',D_max)
print('D_min=',D_min)
print('MCF=',mcf)
print('chromaticity', chromaticity)