#This script contains the standard workflow of DFN inversion

import numpy as np
import scipy
import matplotlib.pyplot as plt
import functions.geometry as geom
import functions.initialize as init
import functions.inversion as inv
import functions.AdvancedHT as HT
import time

#Load observations
Xio = np.load('HT4.npy')

#Initialize inversion
[dim,dx,sx,theta,pdo,frac_l] = init.init()
SR_coord = np.array(((1,0,10),(1,0,15),(1,0,22),(-1,32,5),(-1,32,15),(-1,32,25)))
fracture_intensity = 0.3
fmax = dim[0]*dim[1]*fracture_intensity #Overall fracture length for the domain

#Generate DFN0
DFN0 = inv.RandomDFNGenerator(SR_coord,dim,dx,theta,frac_l,fracture_intensity)
Xi0 = HT.Forward_model(DFN0,SR_coord,theta)
RMS0 = inv.CalculateRMS(Xi0,Xio)
DFN_old = np.copy(DFN0)
phi_old = np.copy(RMS0)
Con = np.array(phi_old)
np.save('results/realization_0',DFN_old)

#rjMCMC control parameters
sigma = 200
P1 = 0.4
P2 = 0.4
P3 = 0.2

#Initialize loop
acc = 0 #Accepted realization number
i = 0
plt.figure()

print('Starting loop...')
start_time = time.time()
while (i <100000):
    update_info_variable = np.zeros(2)
    [P1a,P2a,P3a] = inv.UpdateControl(DFN_old,P1,P2,P3,fmax)
    #Update phase
    update_id = inv.DrawUpdate(P1a,P2a,P3a)
    flag = 0
    trials = 0
    while(flag==0):
        trials = trials+1
        if (update_id==1):
            insertion_points = geom.FindInsertionPoints(DFN_old,dx,dim,theta)
            [DFN_new,flag,selected_ins_id] = geom.AddFracture(DFN_old,insertion_points,frac_l,dx,theta)
            update_info_variable = [selected_ins_id,insertion_points]
        elif(update_id ==2):
            delf = geom.GetDeleteableFracturesP(DFN_old,SR_coord,dim,dx,theta,2)
            [DFN_new,flag,deleted_fracture] = geom.DeleteFracture(DFN_old,delf)
            DFN_new = geom.TagFloat(DFN_new,SR_coord,dim,dx,theta)
            update_info_variable = [delf,deleted_fracture]
        elif(update_id==3):
            #insertion_points = geom.FindInsertionPoints(DFN_old,dx,dim,theta)
            #delf = geom.GetDeleteableFractures(DFN_old,SR_coord,dim,dx,theta)
            [DFN_new,flag] = geom.MoveFracture(DFN_old,SR_coord,dim,dx,theta,frac_l)
            DFN_new = geom.TagFloat(DFN_new,SR_coord,dim,dx,theta)
        if (geom.CheckConnectedDFN(DFN_new,SR_coord,dim,dx,theta,10,20)==0):
            flag = 0
        if ((trials>10) and (flag==0)):
            update_id = inv.DrawUpdate(P1a,P2a,P3a)
    #Evaluation phase
    Xi_new = HT.Forward_model(DFN_new,SR_coord,theta)
    
    [P_forward,P_backward] = inv.CalculateUpdateProbability(update_id,SR_coord,DFN_new,DFN_old,dx,dim,frac_l,theta,P1,P2,P3,update_info_variable)
    
    phi_new = inv.CalculateRMS(Xi_new,Xio)
    
    L_ratio = np.exp((phi_old-phi_new)/(2*sigma**2))
    alpha = min(1,L_ratio*(P_backward/P_forward))
    
    #Decision
    beta = np.random.rand(1)
    if(alpha>beta):
        acc = acc+1
        DFN_old = np.copy(DFN_new)
        phi_old = phi_new
        decision = [phi_new,L_ratio,(P_backward/P_forward)]
        frac_l = inv.UpdateFLD(DFN_old,frac_l,pdo) #Update parameters
        print('Accepted!')
    else:
        DFN_new = np.copy(DFN_old)
        phi_new = phi_old
        decision = [phi_old,L_ratio,(P_backward/P_forward)]
        print('Not accepted!')
    Con = np.append(Con,phi_old)       
    #Save data
    Data = [DFN_old,decision]
    np.save('results/realization_'+str(i),Data)
    #Display
    
    print('phi_new:',phi_new, ' Realization:', i, 'Acceptions:',acc)
    print("--- %s seconds ---" % (time.time() - start_time))
    #Plot
    plt.gcf().clear()
    plt.plot(Con)
    plt.pause(0.01)
    start_time = time.time()
    i = i+1
    
    
    