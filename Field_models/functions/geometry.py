# -*- coding: utf-8 -*-
# This module contains functions for DFN geometry manipulation

import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
#from numba import jit
#from numba import njit
import multiprocessing
from functools import partial
from matplotlib import cm


#Plot stuff
def plot_DFN_field(DFN):
    plt.clf()
    endpoints = 0
    for i in range(0,len(DFN)):
        if (endpoints ==1):
            plt.scatter(DFN[i,0]-0.5*DFN[i,2]*np.cos(DFN[i,3]),DFN[i,1]-0.5*DFN[i,2]*np.sin(DFN[i,3]))
            plt.scatter(DFN[i,0]+0.5*DFN[i,2]*np.cos(DFN[i,3]),DFN[i,1]+0.5*DFN[i,2]*np.sin(DFN[i,3]))
        plt.plot([DFN[i,0]-0.5*DFN[i,2]*np.cos(DFN[i,3]),DFN[i,0]+0.5*DFN[i,2]*np.cos(DFN[i,3])],[DFN[i,1]-0.5*DFN[i,2]*np.sin(DFN[i,3]),DFN[i,1]+0.5*DFN[i,2]*np.sin(DFN[i,3])],'k')
    plt.show()
    plt.pause(0.001)
    return

#Plot stuff
def plot_DFN_field_label(DFN):
    plt.clf()
    endpoints = 0
    for i in range(0,len(DFN)):
        if (endpoints ==1):
            plt.scatter(DFN[i,0]-0.5*DFN[i,2]*np.cos(DFN[i,3]),DFN[i,1]-0.5*DFN[i,2]*np.sin(DFN[i,3]))
            plt.scatter(DFN[i,0]+0.5*DFN[i,2]*np.cos(DFN[i,3]),DFN[i,1]+0.5*DFN[i,2]*np.sin(DFN[i,3]))
        plt.plot([DFN[i,0]-0.5*DFN[i,2]*np.cos(DFN[i,3]),DFN[i,0]+0.5*DFN[i,2]*np.cos(DFN[i,3])],[DFN[i,1]-0.5*DFN[i,2]*np.sin(DFN[i,3]),DFN[i,1]+0.5*DFN[i,2]*np.sin(DFN[i,3])],'k')
        plt.annotate(i,xy=(DFN[i,0],DFN[i,1]))
    #plt.show()
    #plt.pause(0.001)
    return

# The following function identifies possible insertion points in a DFN and computes how long new fractures could be added to their locations
#@jit
#@profile
def FindInsertionPoints(DFN,dx,dim,theta):
    np.seterr(divide='ignore', invalid='ignore')
    insertion_points = np.zeros([0,6])
    err = 0.001
    pos = min(dim)/dx
    for i in range(0,np.shape(DFN)[0]):
        #Get start point of fracture:
        frac_start = [DFN[i,0]-DFN[i,2]/2*np.cos(theta[int(DFN[i,3])]),DFN[i,1]-DFN[i,2]/2*np.sin(theta[int(DFN[i,3])])]
        
        #Get insertion points along the fracture
        stepx = dx*np.cos(theta[int(DFN[i,3])])
        stepy = dx*np.sin(theta[int(DFN[i,3])])
        xy = np.zeros([int(DFN[i,2]/dx)+1,6])
        for j in range(0,int(DFN[i,2]/dx)+1):
            xy[j,0:2] = [frac_start[0]+j*stepx,frac_start[1]+j*stepy]
        
        DFNo = DFN[DFN[:,3] == (3-DFN[i,3]),:]
        #len_v = np.ones(np.shape(DFNo)[0])
        sin_o = np.sin(theta[int(3-DFN[i,3])])
        cos_o = np.cos(theta[int(3-DFN[i,3])])
        
        for j in range(0,np.shape(xy)[0]):
            #Fractures at the same location toward positive direction
            DFNlp = DFNo[np.logical_and(abs(np.sqrt((DFNo[:,0]-xy[j,0])**2 + (DFNo[:,1]-xy[j,1])**2) *np.sin(theta[int(3-DFN[i,3])] - np.arctan((DFNo[:,1]-xy[j,1])/(DFNo[:,0]-xy[j,0])))) < dx/2 , np.logical_or(DFNo[:,0] > xy[j,0] , DFNo[:,1] > xy[j,1])),:]
            #Fractures at the same location toward negative direction
            DFNlm = DFNo[np.logical_and(abs(np.sqrt((DFNo[:,0]-xy[j,0])**2 + (DFNo[:,1]-xy[j,1])**2) *np.sin(theta[int(3-DFN[i,3])] - np.arctan((DFNo[:,1]-xy[j,1])/(DFNo[:,0]-xy[j,0])))) < dx/2 , np.logical_or(DFNo[:,0] < xy[j,0] , DFNo[:,1] < xy[j,1])),:]
            
            #Distance between xy point and fracture endpoint
            val1 = min(np.append(np.sqrt((DFNlp[:,0]-xy[j,0])**2 + (DFNlp[:,1]-xy[j,1])**2)-DFNlp[:,2]/2,pos))/dx
            val2 = min(np.append(np.sqrt((DFNlm[:,0]-xy[j,0])**2 + (DFNlm[:,1]-xy[j,1])**2)-DFNlm[:,2]/2,pos))/dx
            
            #Check for intersections (not valid insertion points)
            EPp = np.array([[DFNlp[:,0]-DFNlp[:,2]/2*cos_o, DFNlp[:,1]-DFNlp[:,2]/2*sin_o],[DFNlp[:,0]+DFNlp[:,2]/2*cos_o, DFNlp[:,1]+DFNlp[:,2]/2*sin_o]])  #Endpoints plus set
            EPm = np.array([[DFNlm[:,0]-DFNlm[:,2]/2*cos_o, DFNlm[:,1]-DFNlm[:,2]/2*sin_o],[DFNlm[:,0]+DFNlm[:,2]/2*cos_o, DFNlm[:,1]+DFNlm[:,2]/2*sin_o]])
            
            if (np.size(EPp) != 0):
                if (np.all([(EPp[0,0] < xy[j,0]),(EPp[0,1] < xy[j,1]),(EPp[1,0] > xy[j,0]),(EPp[1,1] > xy[j,1])])):
                    val1 = 0
                    val2 = 0
                    
            if ((np.size(EPm) != 0) and (val1+val2!=0)):
                if (np.all([(EPm[0,0] < xy[j,0]),(EPm[0,1] < xy[j,1]),(EPm[1,0] > xy[j,0]),(EPm[1,1] > xy[j,1])])):
                    val1 = 0
                    val2 = 0
            
            if ((np.size(DFNo>0)) and (val1+val2!=0)):
                if (((np.min(np.abs(DFNo[:,0]-xy[j,0]))) and (np.min(np.abs(DFNo[:,1]-xy[j,1]))))<err):
                    val1 = 0
                    val2 = 0
#            for k in range(0,np.shape(DFNo)[0]):
#                if ((abs(DFNo[k,0]-xy[j,0]) < err) and (abs(DFNo[k,1]-xy[j,1])<err)):
#                    val1 = 0
#                    val2 = 0
#                    print(k)
#                    break
                    
            #Check boundaries:
            if (sin_o > 0 and cos_o > 0):
                val1 = min([val1,abs((dim[0]-xy[j,1])*(1/sin_o)/dx),abs((dim[1]-xy[j,0])*(1/cos_o)/dx)])
                val2 = min([val2,abs((xy[j,1]-dx)*(1/sin_o)/dx),abs((xy[j,0]-dx)*(1/cos_o)/dx)])
            elif (sin_o > 0 and cos_o < 0):
                val1 = min([val1,abs((dim[0]-xy[j,1])*(1/sin_o)/dx),abs((dx-xy[j,0])*(1/cos_o)/dx)])
                val2 = min([val2,abs((dx-xy[j,1])*(1/sin_o)/dx),abs((dim[1]-xy[j,0])*(1/cos_o)/dx)])
            elif (sin_o < 0 and cos_o > 0):
                val1 = min([val1,abs((dx-xy[j,1])*(1/sin_o)/dx),abs((dim[1]-xy[j,0])*(1/cos_o)/dx)])
                val2 = min([val2,abs((dim[0]-xy[j,1])*(1/sin_o)/dx),abs((xy[j,0]-dx)*(1/cos_o)/dx)])
            else:
                val1 = min([val1,abs((dx-xy[j,1])*(1/sin_o)/dx),abs((dx-xy[j,0])*(1/cos_o)/dx)])
                val2 = min([val2,abs((dim[0]-xy[j,1])*(1/sin_o)/dx),abs((dim[1]-xy[j,0])*(1/cos_o)/dx)])
                
            # Assign rank points after distances
            if (np.size(val1) == 0 or val1 > pos):
                val1 = pos
            if (np.size(val2) == 0 or val2 > pos):
                val2 = pos
                
            xy[j,2] = 3-DFN[i,3]
            xy[j,3] = val1
            xy[j,4] = val2
            xy[j,5] = i
        #add new insertion points
        insertion_points = np.vstack([insertion_points,xy])
    
    #Remove where intersections return negative values:
    insertion_points = np.delete(insertion_points,np.where(insertion_points[:,3] < dx),axis = 0)
    insertion_points = np.delete(insertion_points,np.where(insertion_points[:,4] < dx),axis = 0)
    
    #Remove close to boundaries:
    insertion_points = np.delete(insertion_points,np.where(insertion_points[:,0] <= dx),axis = 0)
    insertion_points = np.delete(insertion_points,np.where(insertion_points[:,1] <= dx),axis = 0)
    insertion_points = np.delete(insertion_points,np.where(insertion_points[:,0] >= dim[1] - dx),axis = 0)
    insertion_points = np.delete(insertion_points,np.where(insertion_points[:,1] >= dim[0] - dx),axis = 0)
    
    return insertion_points     #insertion_points: 0:x 1:y 2:set_ID 3:gap_plus 4:gap_minus 5:fracture_number


#Fracture addition
def AddFracture(DFN,insertion_points,frac_l,dx,theta):
    ID = np.random.randint(1,3) #ID of selected BASE fracture (where we add)
    selected_ins_id = -1
    
    insertion_valid = np.array(np.where(insertion_points[:,2] == 3-ID))[0]
    insertion_valid_set = insertion_points[insertion_valid,:]
    
    if (np.size(insertion_valid) == 0):
        valid = 0
    else:
        valid = 0
        max_iter = 1000
        iterat = 0
        plus = 0
        minus = 0
        while(valid == 0 and iterat<max_iter):
            iterat = iterat+1
            selected_point = np.random.randint(0,np.shape(insertion_valid_set)[0])
            
            xy_insertion = [insertion_valid_set[selected_point,0],insertion_valid_set[selected_point,1]]
            plus = insertion_valid_set[selected_point,3]
            minus = insertion_valid_set[selected_point,4]
            
            #Generate fracture length
            A = np.zeros([1,np.shape(frac_l)[1]])
            while(np.max(A)<=0):
                r2 = np.random.rand(1)
                A = frac_l[2,:]-r2
            l_new = frac_l[0,np.where(A == min(A[A>0]))]
            
            if (ID == 1):
                ID_new = 2
            else:
                ID_new = 1
                
            sel_limit_minus = 1
            sel_limit_plus = 0
            
            sel_limit_minus = 1 + (l_new/dx - (np.floor(plus)-1))*((l_new/dx-(np.floor(plus)-1))>0)
            sel_limit_plus = min([np.floor(minus)+1,l_new/dx+1])
            
            if(np.all([np.floor(minus)+np.floor(plus) > l_new/dx, np.floor(minus) > 0, np.floor(plus) > 0])):
                if (sel_limit_plus-sel_limit_minus <dx/10):
                    sel_new = sel_limit_minus
                else:
                    sel_new = np.random.randint(np.round(sel_limit_minus),np.round(sel_limit_plus))
                
                # Create fracture from segments
                xy_lx = np.linspace(0,np.asscalar(l_new*np.cos(theta[ID_new])),num=int(l_new/dx+1))
                xy_ly = np.linspace(0,np.asscalar(l_new*np.sin(theta[ID_new])),num=int(l_new/dx+1))
                xy_l = np.vstack([xy_lx,xy_ly]).T
                shift = xy_insertion - xy_l[int(sel_new-1),:]
                xy_real = xy_l + np.hstack([np.ones([np.shape(xy_l)[0],1])*shift[0],np.ones([np.shape(xy_l)[0],1])*shift[1]])
                x_center_new = (xy_real[np.shape(xy_real)[0]-1,0]+xy_real[0,0])/2
                y_center_new = (xy_real[np.shape(xy_real)[0]-1,1]+xy_real[0,1])/2
                
                new_line = np.array([x_center_new,y_center_new,l_new,ID_new,1])
                
                DFN = np.vstack([DFN,new_line])
                valid = 1
                slot_size = sel_limit_plus - sel_limit_minus
                selected_ins_idx = np.where(insertion_points[:,0] == xy_insertion[0])
                selected_ins_idy = np.where(insertion_points[:,1] == xy_insertion[1])
                selected_ins_id = np.intersect1d(selected_ins_idx,selected_ins_idy)
            else:
                valid = 0
                slot_size = 1
                
            if (iterat == max_iter):
                valid = 0
    flag = valid
    return DFN,flag,selected_ins_id

#Convert DFN to map - RASTERIZATION - code for visualization purposes only
def MapDFN(DFN,dim,dx,theta):
    Hor = dim[1]
    Ver = dim[0]
    px = dx/10
    py = dx/10
   
    MAP = np.zeros([int(Ver/py)+1,int(Hor/px)+1])
    
    for f in range(0,np.shape(DFN)[0]):
        if DFN[f,2] > np.min([px,py]):
            x1o = DFN[f,0]
            y1o = DFN[f,1]
            x1p = int(np.round(x1o/px))
            y1p = int(np.round(y1o/py))
        
            width_projected = int(np.round(DFN[f,2]*np.cos(theta[int(DFN[f,3])])/px))
            height_projected = int(np.round(DFN[f,2]*np.sin(theta[int(DFN[f,3])])/py))
            
            if (height_projected == 0): 
                height_projected = 1
            if (width_projected == 0):
                width_projected = 1
                
            
            x1s = x1p - np.round(width_projected/2)
            y1s = y1p - np.round(height_projected/2)
        
            steps = np.max([abs(width_projected),abs(height_projected)])
        
            pixels = np.zeros([steps,2])
            pixels[0,:] = [x1s,y1s]
            if (abs(width_projected) > abs(height_projected)):
                filler = 0
                step_x = width_projected/height_projected
                step_y = height_projected/width_projected
                filler = step_y
                for i in range(1,steps):
                    filler = filler + step_y
                    pixels[i,0] = pixels[i-1,0] + 1
                    pixels[i,1] = pixels[0,1] + np.floor(filler)
            else:
                filler = 0
                step_x = width_projected/height_projected
                step_y = height_projected/width_projected
                filler = step_x
                for i in range(1,steps):
                    filler = filler + step_x
                    pixels[i,1] = pixels[i-1,1] + 1
                    pixels[i,0] = pixels[0,0] + np.floor(filler)
          
            pixels = pixels.astype(int)      
        
            for i in range(0,np.shape(pixels)[0]):
                MAP[pixels[i,1],pixels[i,0]] = 1

    return MAP

#Convert DFN to map - RASTERIZATION - with large pixels
# This version of the function is for connectivity checks - with higher resolution, and wider stripes as fractures
def MapDFNLarge(DFN,dim,dx,theta):
    Hor = dim[1]
    Ver = dim[0]
    px = dx
    py = dx
    s = 10  #frame around the DFN (to avoid indexing errors) (all pixels are shifted by this value right and up)
   
    MAP1 = np.zeros([int(Ver/py+2*s)+1,int(Hor/px + 2*s)+1])
    
    for f in range(0,np.shape(DFN)[0]):
        if DFN[f,2] > np.min([px,py]):
            x1o = DFN[f,0]
            y1o = DFN[f,1]
            x1p = int(np.floor(x1o/px+s))
            y1p = int(np.floor(y1o/py+s))
        
            width_projected = int(np.floor(DFN[f,2]*np.cos(theta[int(DFN[f,3])])/px))
            height_projected = int(np.floor(DFN[f,2]*np.sin(theta[int(DFN[f,3])])/py))
            
            if (height_projected == 0): 
                height_projected = 1
            if (width_projected == 0):
                width_projected = 1
                
            
            x1s = x1p - np.floor(width_projected/2)
            y1s = y1p - np.floor(height_projected/2)
        
            steps = np.max([abs(width_projected),abs(height_projected)])
        
            pixels = np.zeros([steps+1,2])
            pixels[0,:] = [x1s,y1s]
            if (abs(width_projected) > abs(height_projected)):
                filler = 0
                step_x = width_projected/height_projected
                step_y = height_projected/width_projected
                #filler = step_y
                for i in range(1,steps+1):
                    filler = filler + step_y
                    pixels[i,0] = pixels[i-1,0] + 1
                    pixels[i,1] = pixels[0,1] + np.floor(filler)
            else:
                filler = 0
                step_x = width_projected/height_projected
                step_y = height_projected/width_projected
                #filler = step_x
                for i in range(1,steps+1):
                    filler = filler + step_x
                    pixels[i,1] = pixels[i-1,1] + 1
                    pixels[i,0] = pixels[0,0] + np.floor(filler)
          
            pixels = pixels.astype(int)      
        
            for i in range(0,np.shape(pixels)[0]):
                MAP1[pixels[i,1],pixels[i,0]] = 1
                MAP1[pixels[i,1]-1,pixels[i,0]] = 1
                MAP1[pixels[i,1]+1,pixels[i,0]] = 1
                MAP1[pixels[i,1],pixels[i,0]-1] = 1
                MAP1[pixels[i,1],pixels[i,0]+1] = 1
                
                MAP1[pixels[i,1]-1,pixels[i,0]-1] = 1
                MAP1[pixels[i,1]+1,pixels[i,0]-1] = 1
                MAP1[pixels[i,1]-1,pixels[i,0]+1] = 1
                MAP1[pixels[i,1]+1,pixels[i,0]+1] = 1
                
    MAP = MAP1[s:np.shape(MAP1)[0]-s,s:np.shape(MAP1)[1]-s]
    return MAP

#Check if sources and receivers are connected:
#@jit#(parallel=True)
def CheckConnectedDFN(DFN,SR_coord,dim,dx,theta,prec,s):
    #prec = 10
    dpx = dx/prec
    #s = 20
    connected = 1
    
    MAP = MapDFNLarge(DFN,dim,dpx,theta)
    label = measure.label(MAP)
    #Select source-receiver pixels
    SRid_pixels = np.zeros([np.shape(SR_coord)[0],2])
    SR_labels = np.zeros(np.shape(SR_coord)[0])
    for i in range(0,np.shape(SR_coord)[0]):
        SRid_pixels[i,0] = int(max(SR_coord[i,1]/dpx-1,0))
        SRid_pixels[i,1] = int(max(SR_coord[i,2]/dpx-1,0))
        SR_labels[i] = label[int(SRid_pixels[i,1]),int(SRid_pixels[i,0])]
        if(i>0):
            if (SR_labels[i] != SR_labels[i-1]):
                connected = 0
    return connected

#Find fractures to delete - where the connection does not break
#@jit(parallel=True)
def GetDeleteableFractures(DFN,SR_coord,dim,dx,theta):
    deleteable_fractures = np.zeros([0,1])
    for i in range(13,np.shape(DFN)[0]):     #This can be parallel
        DFN_minus_one = np.delete(DFN,i,axis=0)
        connected = CheckConnectedDFN(DFN_minus_one,SR_coord,dim,dx,theta,4,10)
        if (connected==1):
            deleteable_fractures = np.vstack([deleteable_fractures,i])
    return deleteable_fractures

#Deleteable fractures - parallel(2functions)
def GDFinside(DFN,SR_coord,dim,dx,theta,i):
    DFN_minus_one = np.delete(DFN,i,axis=0)
    connected = CheckConnectedDFN(DFN_minus_one,SR_coord,dim,dx,theta,4,10)
    a = 0
    if (connected==1):
        a = 1
    return a
    
def GetDeleteableFracturesP(DFN,SR_coord,dim,dx,theta,cores):
    deleteable = np.zeros([np.shape(DFN)[0]])
    pool = multiprocessing.Pool(processes=cores)
    iterator = [i for i in range(13,np.shape(DFN)[0])]
    p = partial(GDFinside,DFN,SR_coord,dim,dx,theta)
    deleteable = pool.map(p,iterator)
    pool.close()
    deleteable_fractures = np.where(deleteable==np.ones(len(deleteable)))+13*np.ones(sum(deleteable)) #shift needed because the removal of the first 12 fractures
    return deleteable_fractures.T

#Delete one fracture
def DeleteFracture(DFN,deleteable_fractures):
    deleted_fracture = np.zeros(0)
    if(np.size(deleteable_fractures)!=0):
        a = np.random.randint(0,np.shape(deleteable_fractures)[0])
        b = deleteable_fractures[a]
        deleted_fracture = DFN[int(b),:]
        DFN = np.delete(DFN,b,axis=0)
        flag = 1
    else:
        flag = 0    
    
    return DFN,flag,deleted_fracture


#Combine addition and deletion
def MoveFracture(DFN,SR_coord,dim,dx,theta,frac_l):
    DFN2 = np.copy(DFN)
    trials = 0
    limit = 20
    valid = 0
    delf = GetDeleteableFracturesP(DFN,SR_coord,dim,dx,theta,2)
    insertion_points = FindInsertionPoints(DFN,dx,dim,theta)
    while (trials<limit):
        trials = trials+1
        [DFN2,val1,x] = DeleteFracture(DFN,delf)
        [DFN2,val2,y] = AddFracture(DFN2,insertion_points,frac_l,dx,theta)
        valid=val1*val2
        if (valid!=0):
            DFN = np.copy(DFN2)
            break
    if (trials >= limit):
        valid = 0
    return DFN,valid
        

           
# Corrected MOVe algorithm
#Modifications: shift along ONE fracture, both sets can move around
#@profile
def MoveFracture_shift(DFN,SR_coord,dim,dx,theta):
    valid = 0
    trial = 0
    limit = 20
    DFN_test_temp = DFN[13:,:]
    insertion_points2 = FindInsertionPoints(DFN_test_temp,dx,dim,theta)
    while(trial<limit):
        DFN_test = np.copy(DFN_test_temp)
        trial = trial + 1
        moved = 0
        # Select a fracture to move
        b = np.random.randint(0,np.shape(DFN_test)[0])
        #P1 = np.array([DFN_test[b,0] - DFN_test[b,2]*0.5*np.cos(theta[int(DFN_test[b,3])]),DFN_test[b,0] - DFN_test[b,2]*0.5*np.sin(theta[int(DFN_test[b,3])])])
        #P2 = np.array([DFN_test[b,0] + DFN_test[b,2]*0.5*np.cos(theta[int(DFN_test[b,3])]),DFN_test[b,0] + DFN_test[b,2]*0.5*np.sin(theta[int(DFN_test[b,3])])])
        Flength = DFN_test[b,2]
        DFN_test2 = np.delete(DFN_test,b,axis=0) #DFN without the selected fracture
        insertion_points = FindInsertionPoints(DFN_test2,dx,dim,theta) #insertion points without
        in_scalar = insertion_points[:,0] + insertion_points[:,1]
        in_scalar2 = insertion_points2[:,0] + insertion_points2[:,1]
        #The difference will give us the intersecting fractures  
        intersect_coord = np.where(np.in1d(in_scalar,in_scalar2)==0)[0]
        
        if ((np.size(intersect_coord)>1)):
            #Now select one point randomly which could work with the selected fracture
            #if (max(intersect_coord) >= np.shape(insertion_points)[0]):
            #    np.delete(intersect_coord,np.where(intersect_coord >= np.shape(insertion_points)[0]))
            order = np.random.permutation(np.size(intersect_coord))
            
            for i in range(0,np.size(intersect_coord)):
                #print()
                #if (intersect_coord[order[i]] <= np.shape(insertion_points)[0]):
                slider_fracture_no = int(insertion_points[intersect_coord[order[i]],5])
                possible_insertions = insertion_points[insertion_points[:,5] == slider_fracture_no,:]
                # Where does the fracture fit:
                slot = possible_insertions[:,3] + possible_insertions[:,4]
                possible_insertions = possible_insertions[np.where(slot > Flength)[0],:]
                if(np.shape(possible_insertions)[0] > 1):
                    #Sort insertion point by distance to fracture
                    d = np.sqrt((insertion_points[intersect_coord[order[i]],0]-possible_insertions[:,0])**2+(insertion_points[intersect_coord[order[i]],1]-possible_insertions[:,1])**2)
                    #Plus or Minus
                    plusorminus = ((np.sign(possible_insertions[:,0]-insertion_points[intersect_coord[order[i]],0]) == np.sign(np.cos(theta[int(3-DFN_test[b,3])]))) + (np.sign(possible_insertions[:,1]-insertion_points[intersect_coord[order[i]],1]) == np.sign(np.sin(theta[int(3-DFN_test[b,3])]))))
                    elojel = np.ones(np.shape(plusorminus)[0])
                    elojel = elojel - 2*(plusorminus==0)
                    
                    elojel = elojel[np.nonzero(d)]
                    d2 = d[np.nonzero(d)]
                    #Gaussian distance selection
                    d_rand = abs(np.random.normal(dx,max(d)/2))
                    d_sel = int(np.argmin(np.abs(d2-d_rand)))
                    
                    
                    #Calculate shift vector
                    shift = elojel[d_sel]*d2[d_sel]
                    shift_x = shift*np.cos(theta[int(3-DFN_test[b,3])])
                    shift_y = shift*np.sin(theta[int(3-DFN_test[b,3])])
                    
                    #Check for conflicts w other  (overlap)
            
                    Xmin_new = min((DFN_test[b,0] - DFN_test[b,2]*0.5*np.cos(theta[int(DFN_test[b,3])]))+shift_x,(DFN_test[b,0] + DFN_test[b,2]*0.5*np.cos(theta[int(DFN_test[b,3])]))+shift_x)
                    Xmax_new = max((DFN_test[b,0] - DFN_test[b,2]*0.5*np.cos(theta[int(DFN_test[b,3])]))+shift_x,(DFN_test[b,0] + DFN_test[b,2]*0.5*np.cos(theta[int(DFN_test[b,3])]))+shift_x)
                    Ymin_new = min((DFN_test[b,1] - DFN_test[b,2]*0.5*np.sin(theta[int(DFN_test[b,3])]))+shift_y,(DFN_test[b,1] + DFN_test[b,2]*0.5*np.sin(theta[int(DFN_test[b,3])]))+shift_y)
                    Ymax_new = max((DFN_test[b,1] - DFN_test[b,2]*0.5*np.sin(theta[int(DFN_test[b,3])]))+shift_y,(DFN_test[b,1] + DFN_test[b,2]*0.5*np.sin(theta[int(DFN_test[b,3])]))+shift_y)
                    
                    Rminus = np.sqrt((Xmin_new - insertion_points[intersect_coord[order[i]],0])**2 + (Ymin_new - insertion_points[intersect_coord[order[i]],1])**2)
                    Rplus = np.sqrt((Xmax_new - insertion_points[intersect_coord[order[i]],0])**2 + (Ymax_new - insertion_points[intersect_coord[order[i]],1])**2) # We use these values later
                    
                    if ((Rminus > insertion_points[intersect_coord[order[i]],4]) & (Rplus > insertion_points[intersect_coord[order[i]],3])):
                        moved = 1
                        #MOVE
                        DFN_test[b,0] = DFN_test[b,0]+shift_x
                        DFN_test[b,1] = DFN_test[b,1]+shift_y
                        break
            
        
    
            #CONTROL
            #Check boundaries
            if(moved==1):
                if((Xmax_new < dim[1]) & (Xmax_new > dx) &
                   (Xmin_new < dim[1]) & (Xmin_new > dx) &
                   (Ymax_new < dim[0]) & (Ymax_new > dx) &
                   (Ymin_new < dim[0]) & (Ymin_new > dx)):
                    DFN_trial = np.copy(DFN)
                    DFN_trial[13:,:] = np.asarray(DFN_test)
                    valid = CheckConnectedDFN(DFN_trial,SR_coord,dim,dx,theta,10,20)
                else:
                    valid = 0
                   
            if(valid ==1):
                DFN = np.copy(DFN_trial)
                break
        
    if (trial>=limit):
        valid = 0
    flag = valid
    return DFN,flag


# Tag floating fractures
def TagFloat(DFN,SR_coord,dim,dx,theta):
    connected = 1
    dpx = dx/10
    
    MAP = MapDFNLarge(DFN,dim,dpx,theta)
    label = measure.label(MAP)
    #Select source-receiver pixels
    SRid_pixels = np.zeros([np.shape(SR_coord)[0],2])
    SR_labels = np.zeros(np.shape(SR_coord)[0])
    for i in range(0,np.shape(SR_coord)[0]):
        SRid_pixels[i,0] = int(max(np.floor(SR_coord[i,1]/dpx),1))
        SRid_pixels[i,1] = int(max(np.floor(SR_coord[i,2]/dpx),1))
        SR_labels[i] = label[int(SRid_pixels[i,1]),int(SRid_pixels[i,0])]
        
    label_true = SR_labels[0]
    
    tag = np.zeros([np.shape(DFN)[0],1])
    for i in range(0,np.shape(DFN)[0]):
        actual_x = int(np.floor(DFN[i,0]/dpx))
        actual_y = int(np.floor(DFN[i,1]/dpx))
        if (int(label[actual_y,actual_x])==int(label_true)):
            tag[i] = 1
        else:
            tag[i] = 0
            
        DFN[i,4] = tag[i]
    return DFN

def CalculateP21(DFN,dim,theta,dx):
    level = 2
    split = 2**level
    
    #Convert to scatter set - each point represent one dx section of DFN
    DFNscatter = np.zeros([0,2])
    for i in range(0,np.shape(DFN)[0]):
        #get start point
        frac_start_x = DFN[i,0] - DFN[i,2]/2*np.cos(theta[int(DFN[i,3])])
        frac_start_y = DFN[i,1] - DFN[i,2]/2*np.sin(theta[int(DFN[i,3])])
        stepx = dx*np.cos(theta[int(DFN[i,3])])
        stepy = dx*np.sin(theta[int(DFN[i,3])])
        steps = int(np.floor(DFN[i,2]/dx))
        
        xy = np.zeros([steps,2])
        for j in range(0,steps):
            xy[j,0] = frac_start_x + j*stepx+stepx/2
            xy[j,1] = frac_start_y + j*stepy+stepy/2
        DFNscatter = np.vstack([DFNscatter,xy])
    
    P = np.zeros([split,split])    
    for i in range(0,split):
        for j in range(0,split):
            Xbox1 = i*dim[1]/split
            Xbox2 = (i+1)*dim[1]/split
            Ybox1 = j*dim[0]/split
            Ybox2 = (j+1)*dim[0]/split
            for k in range(0,np.shape(DFNscatter)[0]):
                if((DFNscatter[k,0] >= Xbox1) & (DFNscatter[k,0] <= Xbox2) & (DFNscatter[k,1] >= Ybox1) & (DFNscatter[k,1] <= Ybox2)):
                    P[i,j] = P[i,j] + 1
                    
    area = (Xbox2-Xbox1)*(Ybox2-Ybox1)
    
    P21 = P/area
    return P21
#Scripts
#insertion_points = FindInsertionPoints(DFN,dx,dim,theta)
#DFN1 = AddFracture(DFN1,insertion_points,frac_l,dx,theta)[0]
#DFN2 = MoveFracture(DFN2,dim,dx,theta)[0]
#
#plt.figure()
#plt.imshow(MAP)
#for i in range(0,np.shape(DFN)[0]):
#    #plt.scatter(np.floor((DFN[i,0]+DFN[i,2]*0.5*np.cos(theta[DFN[i,3].astype(int)]))/dpx),np.floor((DFN[i,1]+DFN[i,2]*0.5*np.sin(theta[DFN[i,3].astype(int)]))/dpx))
#    plt.scatter(np.floor(DFN[i,0]/dpx),np.floor(DFN[i,1]/dpx))
    
def plot_heat(DFN_Mat,T_new):
    for t in range(0,100):
        for i in range(0,3):
            plt.subplot(1,3,i+1)    
            plt.scatter(DFN_Mat[:,1],DFN_Mat[:,2],c = (T_new[i][t,:]), cmap = 'jet')
        plt.show()
        plt.pause(0.01)
        plt.title('time '+str(t))

        