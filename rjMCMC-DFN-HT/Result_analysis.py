# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import functions.AdvancedHT as HT
import functions.initialize as init
import functions.inversion as inv
import functions.geometry as geom

import multiprocessing
import time


[dim,dx,sx,theta,pdo,frac_l] = init.init()
SR_coord = np.array(((1,0,15),(1,0,22),(1,0,29.7),(-1,51,5.3),(-1,51,20.5),(-1,51,29.5)))

start = 10000
iterations = 10000
folder = 'results_1'

#load data into a list
Data_list = []

for i in range(start,start+iterations):
    Data = np.load(folder+'/realization_'+str(i)+'.npy')
    DFN = Data[0]
    Data_list.append(DFN)
    
#Get convergence data
Con = np.zeros([0,3])  
for i in range(0,start+iterations):
    Data = np.load(folder+'/realization_'+str(i)+'.npy')
    Con = np.vstack([Con,Data[1]])
    
thin = 10
prec = 8
MAPcube = np.zeros([int(iterations/thin),int(dim[0]*prec/dx+1),int(dim[1]*prec/dx+1)])
for i in range(start,start+iterations,thin):
    print(str((i-start)/iterations*100) + '%')
    DFN = Data_list[i-start]
    #[T,DFN2] = SFM.Forward_model(DFN,SR_coord,theta,1)
    #DFN3 = np.vstack([DFN2[0,:,:],DFN2[1,:,:],DFN2[2,:,:]])
    Map = geom.MapDFNLarge(DFN,dim,dx/prec,theta)
    MAPcube[int((i-start)/thin),:,:] = Map
FPM = np.nanmean(MAPcube,0)
plt.figure()
plt.imshow(FPM*(FPM>0.1))
#plt.gray()
plt.set_cmap('gray_r')
plt.gca().invert_yaxis()
#plt.savefig('FPM_09.png')
