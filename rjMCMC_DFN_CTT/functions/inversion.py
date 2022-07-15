# -*- coding: utf-8 -*-

import numpy as np
import rjMCMC_DFN_CTT.functions.geometry as geom
import scipy



def RandomDFNGenerator(SR_coord,dim,dx,theta,frac_l,fracture_intensity):
    # Random DFN generator
    connected = 0
    con = 5     #Connector fracture
    err = 10^-10
    
    
    fx = dx/2
    
    #DFN definition: 1:Xcenter 2:Ycenter 3:Length 4:Set_ID
    
    #Buildup source-receiver fractures
    DFN = np.zeros([12,5])
    DFN[0:6,0:2] = SR_coord[:,1:3]
    DFN[0:6,2] = DFN[0:6,2]+1*dx
    DFN[0:6,3] = DFN[0:6,3]+1
    DFN[:,4] = DFN[:,4]+1
    
    #Add connector fractures
    for i in range(6,9):
        DFN[i,0] = DFN[i-6,0]+fx*np.cos(theta[1])+con/2*np.cos(theta[1])
        DFN[i,1] = DFN[i-6,1]+fx*np.sin(theta[1])+con/2*np.sin(theta[1])
        DFN[i,2] = con
        DFN[i,3] = 1
        
    for i in range(9,12):
        DFN[i,0] = DFN[i-6,0]-fx*np.cos(theta[1])-con/2*np.cos(theta[1])
        DFN[i,1] = DFN[i-6,1]-fx*np.sin(theta[1])-con/2*np.sin(theta[1])
        DFN[i,2] = con
        DFN[i,3] = 1
        
    
    #Grow DFN tree
    trials = 0
    DFNstart = np.copy(DFN)
    number = 500
    while((connected==0) & (trials < 3)):
        DFN2 = np.copy(DFNstart)
        for i in range(0,number):
            insertion_points = geom.FindInsertionPoints(DFN2[6:,:],dx,dim,theta)    #Fill from both sides
            if(sum(DFN2[:,2])<2500):
                DFN2 = geom.AddFracture(DFN2,insertion_points,frac_l,dx,theta)[0]
            else:
                break
        connected = geom.CheckConnectedDFN(DFN2,SR_coord,dim,dx,theta,8,20)
        trials = trials+1
        number = number+100
        
    DFN = np.copy(DFN2)
    #Devolve DFN
    P21 = geom.CalculateP21(DFN,dim,theta,dx)
    Pstar = np.ones(np.shape(P21))*fracture_intensity
    sigmaold = sum(sum(abs(Pstar-P21)))
    stop_criteria = fracture_intensity/10#sigmaold/100
    for i in range(0,2*number):
        DFN3 = np.copy(DFN2)
        pick = np.random.randint(13,np.shape(DFN3)[0])
        DFN3 = np.delete(DFN3,pick,axis=0)
        P21 = geom.CalculateP21(DFN3,dim,theta,dx)
        sigmanew = sum(sum(abs(Pstar-P21)))
        connected = geom.CheckConnectedDFN(DFN3,SR_coord,dim,dx,theta,8,20)
        if ((connected == 1) & (sigmanew < sigmaold)):
            DFN2 = np.copy(DFN3)
            sigmaold = sigmanew
            if sigmaold < stop_criteria:
                break
    DFN = DFN2
    return DFN

#Draw random update 1-2-3
def DrawUpdate(P1,P2,P3):
    #check if probabilites add up to 1
    if (P1+P2+P3 != 1):
        Psum = P1+P2+P3
        P1 = P1/Psum
        P2 = P2/Psum
        P3 = P3/Psum
        
    r1 = np.random.random()
    if (r1 <= P1):
        update = 1
    elif (r1<= P1+P2):
        update = 2
    else:
        update = 3
        
    return update

#refresh Fracture Length distribution to eliminate distortions
def UpdateFLD(DFN,frac_l,pdo):
    Pfld_actual = scipy.stats.norm.fit(DFN[:,2])
    
    mu3 = 2*pdo[0] - Pfld_actual[0]
    sigma3 = pdo[1]**2/Pfld_actual[1]
    frac_l[1,:] = 1/(sigma3*np.sqrt(2*np.pi))*np.exp(-((frac_l[0,:]-mu3)**2)/(2*sigma3**2))  #FLD PDF
    
    
    for i in range(0,np.shape(frac_l)[1]):
        frac_l[2,i] = sum(frac_l[1,0:i])   
        
    return frac_l
    
def CalculateRMS(Xi,Xio):
    RMS = np.sqrt(np.sum(np.sum((Xi-Xio)**2)))
    return RMS

def UpdateControl(DFN,P1,P2,P3,fmax):
    length = sum(DFN[:,2])
    if(length > fmax + 0.2*fmax):    #120% length range allowed
        P1 = 0
    elif(length <fmax - 0.2*fmax):
        P2 = 0
    return P1,P2,P3
            
def CalculateUpdateProbability(update_id,SR_coord,DFN_new,DFN_old,dx,dim,frac_l,theta,P1,P2,P3,update_info_variable):
    #ADDITION
    if(update_id == 1):
        #Forward step
        new_fracture = DFN_new[np.shape(DFN_new)[0]-1,:] #Last line in DFN
        insertion_points = update_info_variable[1]
        insertion_id = update_info_variable[0][0]
        
        P_xy = 1/np.shape(insertion_points)[0]
        P_length = frac_l[1,frac_l[0,:] == new_fracture[2]]  #More complicated in MATLAB - beckward step need to be controlled!!!
        P_pos = 1/((np.floor(insertion_points[insertion_id,3]) + np.floor(insertion_points[insertion_id,4]) - new_fracture[2]/dx-1))
        P_forward = P1*P_xy*P_length*P_pos
        
        #Backward step
        delf = geom.GetDeleteableFracturesP(DFN_new,SR_coord,dim,dx,theta,2)
        P_backward = P2*1/np.shape(delf)[0]
    
    #DELETION
    if(update_id == 2):
        #Forward step
        delf = update_info_variable[0]
        P_forward = P2*1/np.shape(delf)[0]
        #Backward step
        deleted_fracture = update_info_variable[1]
        insertion_points = geom.FindInsertionPoints(DFN_new,dx,dim,theta)
        P_xy = 1/np.shape(insertion_points)[0]
        P_length = np.float(frac_l[1,frac_l[0,:] == deleted_fracture[2]])
        P_pos = 1/(deleted_fracture[2]/dx)
        P_backward = P1*P_xy*P_length*P_pos
    if(update_id == 3):
        P_forward = 1
        P_backward = 1
    return P_forward,P_backward
