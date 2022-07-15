# -*- coding: utf-8 -*-
# This script loads a mapped DFN and simulates a forward model on it

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import functions.geometry as geom
import functions.initialize as init
#import functions.FieldForwardModel as FFM
import functions.AdvancedFieldForwardModel as AFFM
import functions.AdvancedFieldForwardModelTTT as AFFMTTT
import functions.HydraulicTomography as HT

[dim,dx,sx,theta,pdo,frac_l] = init.init()
SR_coord = np.array(((1,0,10),(1,0,15),(1,0,22),(-1,32,5),(-1,32,15),(-1,32,25)))
#DFN structure:
#0 - Xcenter, 1 - Ycenter, 2 - Length, 3 - Angle (radian), 
#DFN_data = scipy.io.loadmat('models/DFN_field_file.mat')
#DFN = DFN_data['DFN']
DFN = np.load('models/DFN4.npy')

#Assign fracture sets: #4-set
#plt.hist(DFN[:,3],100)
sets = np.ones(np.shape(DFN)[0])
for i in range(0,np.shape(DFN)[0]):
    if DFN[i,3]<0:  #limit angle of sets
        sets[i] = 2
DFN = np.column_stack([DFN,sets])
        

Xi = AFFM.Forward_model(DFN,SR_coord)
XiT = AFFMTTT.Forward_model(DFN,SR_coord)

#Store
#np.save('observations2/CTT4.npy',Xi)

