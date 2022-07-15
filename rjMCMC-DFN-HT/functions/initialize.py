# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def init():
    dim = [30,32]   #Domain dimensions [xmax,ymax]
    
    dx = 0.5  #Grid resolution
    sx = 1  #Percolation length
    
    theta = [0,-0.34,1.3] #Set inclinations [0,set1,set2]
        
    #Fracture length distribution
    frac_l = np.zeros([3,int(min(dim)//dx-1)])
    frac_l[0,:] = list(np.arange(dx,min(dim),dx))   #Fracture size bins
    
    #Distribution properties (normal pdf):
    sigma = 8.5
    mu = 9.9
    pdo = [mu,sigma]
    
    frac_l[1,:] = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((frac_l[0,:]-mu)**2)/(2*sigma**2))  #FLD PDF
    
    
    for i in range(0,np.shape(frac_l)[1]):
        frac_l[2,i] = sum(frac_l[1,0:i])    #FLD CDF

    return dim,dx,sx,theta,pdo,frac_l