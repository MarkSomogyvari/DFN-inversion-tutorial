# -*- coding: utf-8 -*-
#   v1.1
#Function version of forward model with tracer transport

import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit


@jit(parallel=True)
#@profile
def DFN_transport(DFN,SR_coord,theta):  
    #Function part starts here
    #inputs: DFN + Source-Receiver locations(SR_coord)
    #Original by dr. Mohammadreza Jalali (ETH Zurich)
    #Returns: Tobs 3-D matrix: [:,.,.] - injection_ID, [.,:,.] - time, [.,.,:] - observation point
    
    DFN_Coord_File = DFN[DFN[:,4]==1,:]
    SR_Coord_File = SR_coord
    
    #scipy.sparse.linalg.use_solver(useUmfpack=1,assumeSortedIndices=1)
    
#    theta1 = np.radians(-12)
#    theta2 = np.radians(58.9)
#    theta = [0,theta1,theta2]
    
    
    #Input parameters
    gs = 1          # Limit size to divide up larger fractures (discretization length)
    eps = 0.1       # Maximum allowed error for the connectivity matrix
    Visc = 0.001    # Fluis viscosity [Pa.s]
    cf = 1.0e-9     # Fluid compressibility [1/Pa]
    Aper_1 = 0.0015 # Fracture aperture [m]
    Aper_2 = 0.001
    Dt_P = 0.5      # Pressure time step for steady state [sec]
    DP_time = 2500
    n_time_P = int(DP_time/Dt_P) # Maximum number of pressure time steps
    P_Inj = 3e5     # Injection pressure [Pa]
    P_Prod = 1e5    # Production pressure [Pa]
    P_ini = 2e5     # Initial pressure in the fractures [Pa]
    Dt_T = 20       # Temperature time step [sec]
    DT_time = 2000  # Cooling/Warming time [sec]
    K_T = 0.6       # Thermal conductivity of water @ 20 C [J/m.s.C]
    Rho_f = 1000    # Density of water [kg/m3]
    Cf = 420        # Specific heat of water [J/kg.C]
    T_ini = 20      # Initial temperature of fractures [C]
    T_Inj = 40      # Injection temperature
    epsl = 10 # Convergence criteria ???
    Kapa = K_T/(Rho_f*Cf)
    
    # Create the New Fracture Coordinate and Connectivity Matrix
    # DFN_mat = [0-set_id ,1-xc, 2-yc, 3-length, 4-angle, $$$5-set_id$$$, 5-xbeg, 6-ybeg, 7-xend, 8-yend]
    # Connect_Mat: [i1 i2 i3 i4 i5 i6]
    
    Frac_Seg_Mat = np.zeros([np.shape(DFN_Coord_File)[0],9])
    for i in range(0,np.shape(DFN_Coord_File)[0]):
        Frac_Seg_Mat[i,:] = [DFN_Coord_File[i,3], DFN_Coord_File[i,0], DFN_Coord_File[i,1], DFN_Coord_File[i,2], theta[int(DFN_Coord_File[i,3])],
                     DFN_Coord_File[i,0] - DFN_Coord_File[i,2]/2*np.cos(theta[int(DFN_Coord_File[i,3])]), DFN_Coord_File[i,1] - DFN_Coord_File[i,2]/2*np.sin(theta[int(DFN_Coord_File[i,3])]),
                     DFN_Coord_File[i,0] + DFN_Coord_File[i,2]/2*np.cos(theta[int(DFN_Coord_File[i,3])]), DFN_Coord_File[i,1] + DFN_Coord_File[i,2]/2*np.sin(theta[int(DFN_Coord_File[i,3])])]
        
    # Segmentation between intersections
    Segments = np.array([[],[],[],[],[],[],[]]).T
    coord_seg_start = Frac_Seg_Mat[:,5:7].T
    coord_seg_end = Frac_Seg_Mat[:,7:9].T
    
    for i in range(0,np.shape(Frac_Seg_Mat)[0]):
        # Rotate all fractures so that the processed fracture is horizontal and centered at (0,0)
        seg_orient = Frac_Seg_Mat[i,4]
        Rot_Mat = np.array([[np.cos(seg_orient), np.sin(seg_orient)],[-np.sin(seg_orient), np.cos(seg_orient)]]) 
        #Coord of the start point of rotated fractures:
        ### coord_seg_start_rot = np.dot((coord_seg_start - np.matlib.repmat(Frac_Seg_Mat[i,1:3],np.shape(coord_seg_start[:,0])[0],1)),Rot_Mat)
        coord_seg_start_rot = np.dot(Rot_Mat,(coord_seg_start - np.matlib.repmat(Frac_Seg_Mat[i,1:3],np.shape(coord_seg_start)[1],1).T))    #some rounding error appear here!!
        #Coord of the end point of rotated fractures:
        coord_seg_end_rot = np.dot(Rot_Mat,(coord_seg_end - np.matlib.repmat(Frac_Seg_Mat[i,1:3],np.shape(coord_seg_end)[1],1).T))
        
        # Index of fractures intersecting the y=0 line (in rotated axes)
        idx_inter = np.where((coord_seg_start_rot[1,:]*coord_seg_end_rot[1,:]) <= 0)
        # Remove the currently processed fracture from the list:
        idx_inter = np.setdiff1d(idx_inter,i)
        
        # X-coordinate of intersection of going-through fracture with the y=0 line
        x_inter = coord_seg_end_rot[1,idx_inter]/(coord_seg_end_rot[1,idx_inter] - coord_seg_start_rot[1,idx_inter])*(
                coord_seg_start_rot[0,idx_inter] - coord_seg_end_rot[0,idx_inter]) + coord_seg_end_rot[0,idx_inter]
        
        # Indices of intersections with the fracture range (actual intersection with the fracture)
        idx_x_int = np.intersect1d(np.where(x_inter>-Frac_Seg_Mat[i,3]/2), np.where(x_inter<Frac_Seg_Mat[i,3]/2))
        
        #Points along fractures (including ends and intersections in rotated frame)
        if (np.size(idx_x_int) !=0):
            point_seg_rot = np.hstack([-Frac_Seg_Mat[i,3]/2,np.sort(x_inter[idx_x_int]),Frac_Seg_Mat[i,3]/2])
        else:
            point_seg_rot = [-Frac_Seg_Mat[i,3]/2,Frac_Seg_Mat[i,3]/2]
        
        #Rotate back
        Rot_Mat = np.array([[np.cos(seg_orient), -np.sin(seg_orient)],[np.sin(seg_orient), np.cos(seg_orient)]]) 
        #Points along joints (including ends and intersections) unrotated
        point_seg = np.dot(Rot_Mat,np.vstack([point_seg_rot,np.zeros(np.size(point_seg_rot))])).T + np.matlib.repmat(Frac_Seg_Mat[i,1:3].T,np.size(point_seg_rot),1)
        
        # Segments matrix
        # 0-sx1,1-sy1,2-sx2, 3-sy2, 4-slength, 5-sangle, 6-set
        for j in range(0,np.size(point_seg[:,0])-1):
            newline = np.array([point_seg[j,0], point_seg[j,1], point_seg[j+1,0], point_seg[j+1,1], point_seg_rot[j+1]-point_seg_rot[j], Frac_Seg_Mat[i,4], Frac_Seg_Mat[i,0]])
            Segments = np.vstack([Segments,newline])
            
    #Miniplot
    #for i in range(0,np.shape(Segments[:,1])[0]):
    #    plt.plot([Segments[i,0],Segments[i,2]],[Segments[i,1],Segments[i,3]])
    
            
    # Redistribute data
    # 0-nset, 1-xc, 2-yc, 3-gridsize, 4-angle, 5-xb, 6-yb, 7-xe, 8-ye
    DFN_Mat = np.zeros([np.shape(Segments)[0],9])
    DFN_Mat[:,0] = Segments[:,6]
    DFN_Mat[:,1] = (Segments[:,0]+Segments[:,2])/2
    DFN_Mat[:,2] = (Segments[:,1]+Segments[:,3])/2
    DFN_Mat[:,3] = Segments[:,4]
    DFN_Mat[:,4] = Segments[:,5]
    DFN_Mat[:,5] = Segments[:,0]
    DFN_Mat[:,6] = Segments[:,1]
    DFN_Mat[:,7] = Segments[:,2]
    DFN_Mat[:,8] = Segments[:,3]
    
    
    
      
    #Subdivide fracture segments
    DFN_Mat2 = np.array([[],[],[],[],[],[],[],[],[]]).T
    for i in range(0,np.shape(DFN_Mat)[0]):
        if DFN_Mat[i,3] <=gs:
            DFN_Mat2 = np.vstack([DFN_Mat2,DFN_Mat[i,:]])
        else:
            n_seg = np.floor(DFN_Mat[i,3]/gs)
            d_seg = DFN_Mat[i,3]/n_seg
            for j in range(0,int(n_seg)):
                newline0 = DFN_Mat[i,0]
                newline1 = DFN_Mat[i,5]+(j-0.5)*d_seg*np.cos(DFN_Mat[i,4])
                newline2 = DFN_Mat[i,6]+(j-0.5)*d_seg*np.sin(DFN_Mat[i,4])
                newline3 = d_seg
                newline4 = DFN_Mat[i,4]
                newline5 = DFN_Mat[i,5]+j*d_seg*np.cos(DFN_Mat[i,4])
                newline6 = DFN_Mat[i,6]+j*d_seg*np.sin(DFN_Mat[i,4])
                newline7 = DFN_Mat[i,5]+(j+1)*d_seg*np.cos(DFN_Mat[i,4])
                newline8 = DFN_Mat[i,6]+(j+1)*d_seg*np.sin(DFN_Mat[i,4])
                DFN_Mat2 = np.vstack([DFN_Mat2,[newline0,newline1,newline2,newline3,newline4,newline5,newline6,newline7,newline8]])
                
    DFN_Mat = DFN_Mat2
    
    DFN_Mat = np.delete(DFN_Mat,np.where(DFN_Mat[:,3]<eps),axis =0) # Remove very small fracture segments (artifacts)
    
    #Miniplot2
    #for i in range(0,np.shape(DFN_Mat[:,1])[0]):
    #    plt.plot([DFN_Mat[i,5],DFN_Mat[i,7]],[DFN_Mat[i,6],DFN_Mat[i,8]])
    
    #Connectivity matrix
    Connect_Mat = np.ones([np.shape(DFN_Mat[:,1])[0],6])*(-999)   #fixed in v1.1
    
    for i in range(0,np.size(DFN_Mat[:,1])):
        D55 = abs(DFN_Mat[:,5]-DFN_Mat[i,5])
        D66 = abs(DFN_Mat[:,6]-DFN_Mat[i,6])
        D77 = abs(DFN_Mat[:,7]-DFN_Mat[i,7])
        D88 = abs(DFN_Mat[:,8]-DFN_Mat[i,8])
        D57 = abs(DFN_Mat[:,7]-DFN_Mat[i,5])
        D68 = abs(DFN_Mat[:,8]-DFN_Mat[i,6])
        D75 = abs(DFN_Mat[:,5]-DFN_Mat[i,7])
        D86 = abs(DFN_Mat[:,6]-DFN_Mat[i,8])
        index = np.where(((D55<=eps) & (D66<=eps)) | ((D77<=eps) & (D88<=eps)) | ((D57<= eps) & (D68<=eps)) | ((D75<=eps) & (D86<=eps)))
        k = 0
        for j in range(0,min(5,len(index[0]))):
            if (index[0][j]!=i):
                Connect_Mat[i,k] = index[0][j]
                k = k+1
                
    #Debug DFN_Mat using the connectivity matrix
    badsector = np.array(np.where(Connect_Mat[:,0]==-999)).T
    DFN_Mat = np.delete(DFN_Mat,badsector, axis=0)
    #Correct Con matrix again
    for i in range(0,np.size(badsector)):
        Connect_Mat = np.delete(Connect_Mat,badsector[i], axis=0)
        Connect_Mat = np.delete(Connect_Mat,np.where(Connect_Mat == badsector[i]),axis = 0)
        Connect_Mat[np.where(Connect_Mat > badsector[i])] = Connect_Mat[np.where(Connect_Mat > badsector[i])] - np.ones(np.shape(Connect_Mat[np.where(Connect_Mat > badsector[i])]))
        badsector = badsector - np.ones([np.shape(badsector)[0],1])
    
    
    # Select Source-Receiver points
    ninj = 0
    nprod = 0
    Inj_id = np.zeros(3)
    Prod_id = np.zeros(3)
    DFN_coords = DFN_Mat[:,1:3]
    for i in range(0,np.shape(SR_Coord_File)[0]):
        id1 = scipy.spatial.cKDTree(DFN_coords).query(SR_Coord_File[i,1:3])[1]   #Select closest point in DFN to the SR point
        if (SR_Coord_File[i,0]==1):
            Inj_id[ninj] = id1
            ninj = ninj + 1
        elif (SR_Coord_File[i,0]==-1):
            Prod_id[nprod] = id1
            nprod = nprod + 1
            
    # Estimate the Coefficient of the Pressure Diffusion PDE
    # Create Conductivity & Storativity Matrix
    num_Frac = np.shape(DFN_Mat)[0]
    AP = np.zeros([num_Frac,num_Frac])  # Init pressure conductivity Matrix(LHS)
    BP = np.zeros(num_Frac) # Init pressure storativity Vector (RHS)
    
    Gamma_P = np.zeros(num_Frac)
    Gamma_P_tot = np.zeros(num_Frac)
    Etha_P = np.zeros(num_Frac)
    
    for i in range(0,num_Frac):
        if (DFN_Mat[i,0] == 1):
            Gamma_P[i] = Dt_P*Aper_1**3/(6*Visc*DFN_Mat[i,3])    # Transmissivity 1
            Etha_P[i] = Aper_1*cf*DFN_Mat[i,3]                  # Storativity 1
        else:
            Gamma_P[i] = Dt_P*Aper_2**3/(6*Visc*DFN_Mat[i,3])    # Transmissivity 2
            Etha_P[i] = Aper_2*cf*DFN_Mat[i,3]                  # Storativity 2
            
    for i in range(0,num_Frac):
        Gamma_P_tot[i] = Gamma_P[i]
        for j in range(0,6):
            cn = int(Connect_Mat[i,j])
            if (cn >= 0):
                Gamma_P_tot[i] = Gamma_P_tot[i] + Gamma_P[cn]   # if and g1g2 based on index
                AP[i,cn] = Gamma_P[i]*Gamma_P[cn]
                AP[i,i] = AP[i,i]-Gamma_P[i]*Gamma_P[cn]
        AP[i,i] = AP[i,i]/Gamma_P_tot[i] - Etha_P[i]
        for j in range(0,6):
            cn = int(Connect_Mat[i,j])
            if (cn >= 0):
                AP[i,cn] = AP[i,cn]/Gamma_P_tot[i]
                
    #for idinj in range(0,3) # Loop for injection location
    Tobs = np.zeros([3,int(DT_time/Dt_T)+1,3])
    for idinj in range(0,3):
    
        #Solve for steady-state pressure
        tp = -1
        #timeP = np.linspace(Dt_P,DP_time,n_time_P)
        P_old = np.ones(num_Frac)
        P_new = np.empty([n_time_P,num_Frac])
        #P_new = [0]*num_Frac
        BP = np.zeros(num_Frac)
        APl = np.copy(AP) #Local AP variable (loopwise)
        
        # Solve Finite Difference pressure
        while (tp < n_time_P-1):
            tp = tp + 1
            
            #Apply initial conditions
            if (tp==0):
                P_old = P_old*P_ini     # Initial pressure
            else:
                P_old[0:num_Frac] = P_new[tp-1,:]
                
#            for i in range(0,num_Frac):
#                BP[i] = -Etha_P[i]*P_old[i]
            BP = -Etha_P*P_old
            
                
            # Apply boundary conditions
            # Injection points
            APl[int(Inj_id[idinj]),:] = 0
            APl[int(Inj_id[idinj]),int(Inj_id[idinj])] = 1
            BP[int(Inj_id[idinj])] = P_Inj
            #Production points
            for i in range(0,np.shape(Prod_id)[0]):
                APl[int(Prod_id[i]),:] = 0
                APl[int(Prod_id[i]),int(Prod_id[i])] = 1
                BP[int(Prod_id[i])] = P_Prod
                
            # Solver for pressure
            #AP2 =  #SPARSE MATRIX NEEDED for performance
            AP2 = scipy.sparse.csc_matrix(APl)
            
            #P_new[tp,:] = scipy.linalg.solve(AP,BP.T)
            P_new[tp,:] = scipy.sparse.linalg.spsolve(AP2,BP.T,permc_spec='MMD_ATA')
#            error = 0
#            for i in range(0,num_Frac):
#                change = abs(P_new[tp,i]-P_old[i])
#                if ( change > error):
#                    error = change    #EZT LE KELL ELLENORIZNI + EGYSZERUSITENI!!!
            error = np.max(abs(P_new[tp,:]-P_old))
            if error < epsl:
                break
        tss = tp
        
        P_new = P_new[0:tss+1,:]
        
        #Estimate steady-state velocity
        num_up = np.zeros(num_Frac)
        V = np.zeros(num_Frac)
        for i in range(0,num_Frac):
            P_Max = P_new[tss,i]
            num_up[i] = i
            for j in range(0,6):
                cn = int(Connect_Mat[i,j])
                if (cn >= 0 and P_new[tp,cn] > P_Max):
                    P_Max = P_new[tp,cn]
                    num_up[i] = cn
            
            if (DFN_Mat[i,0] == 1):
                V[i] = Aper_1**2/(12*Visc*DFN_Mat[i,3])*(P_Max-P_new[tp,i])
            else:
                V[i] = Aper_2**2/(12*Visc*DFN_Mat[i,3])*(P_Max-P_new[tp,i])
                
        #Plot velocity
        
             
        # Estimate the Coefficient of the Temperature Diffusion PDE
        # Create Conductivity & Storativity Matrix
        Gamma_T = np.zeros(num_Frac)
        XI_T = np.zeros(num_Frac)
        Etha_T = np.zeros(num_Frac)
        for i in range(0,num_Frac):
            if (DFN_Mat[i,0] == 1):
                Gamma_T[i] = 2*Dt_T*Kapa*Aper_1/DFN_Mat[i,3]
                XI_T[i] = Aper_1*Dt_T*V[i]
                Etha_T[i] = Aper_1*DFN_Mat[i,3]
            else:
                Gamma_T[i] = 2*Dt_T*Kapa*Aper_2/DFN_Mat[i,3]
                XI_T[i] = Aper_2*Dt_T*V[i]
                Etha_T[i] = Aper_2*DFN_Mat[i,3]
                
        #Initialize temperature matrices
        AT = np.zeros([num_Frac,num_Frac])    #Conductivity matrix(LHS)
        BT = np.zeros(num_Frac)     #Storativity vector(RHS)
        #Gamma_T_tot = np.zeros(num_Frac)
        Gamma_T_tot = np.array(Gamma_T)
        for i in range(0,num_Frac):
            #Gamma_T_tot[i] = Gamma_T[i]
            for j in range(0,5):
                cn = int(Connect_Mat[i,j])
                if (cn>=0):
                    Gamma_T_tot[i] = Gamma_T_tot[i]+Gamma_T[cn]
                    AT[i,cn] = Gamma_T[i]*Gamma_T[cn]
                    AT[i,i] = AT[i,i]-(Gamma_T[i]*Gamma_T[cn])
            AT[i,i] = AT[i,i]/Gamma_T_tot[i]-Etha_T[i]
            for j in range(0,5):
                cn = int(Connect_Mat[i,j])
                if (cn>=0):
                    AT[i,cn] = AT[i,cn]/Gamma_T_tot[i]
            
            AT[i,i] = AT[i,i]-XI_T[i]
            AT[i,int(num_up[i])] = AT[i,int(num_up[i])]+XI_T[i]
            
        #Solve for temperature
        tT = -1
        n_time_T = int(DT_time/Dt_T)
        #time = np.linspace(Dt_T,DT_time,n_time_T)
        T_new = np.zeros([n_time_T+1,num_Frac]) 
        
        #Solve Finite Difference Temperature
        while (tT < n_time_T):
            tT = tT + 1
            if (tT==0):
                T_old = np.ones(num_Frac)*T_ini
            else:
                T_old = T_new[tT-1,:]
                
#            for i in range(0,num_Frac):
#                BT[i] = -Etha_T[i]*T_old[i]
            BT = -Etha_T * T_old
                
            #Apply boundary conditions
            AT[int(Inj_id[idinj]),:] = 0
            AT[int(Inj_id[idinj]),int(Inj_id[idinj])] = 1
            BT[int(Inj_id[idinj])] = T_Inj
            
            #Solver
            #AT2 = scipy.sparse.coo_matrix(AT).tocsc()
            #AT2 = scipy.sparse.csc_matrix((num_Frac,num_Frac))
            AT2 = scipy.sparse.csr_matrix(AT)
            
            #T_new[tT,:] = scipy.sparse.linalg.spsolve(AT2,BT.T,permc_spec="COLAMD",use_umfpack = 'True')
            #T_new[tT,:] = scipy.sparse.linalg.bicgstab(AT2,BT.T,x0 = T_new[tT-1,:],tol=1e-5)[0]
            T_new[tT,:] = scipy.sparse.linalg.spsolve(AT2,BT.T,permc_spec='MMD_AT_PLUS_A',use_umfpack=0)
            #T_new[tT,:] = scipy.linalg.solve(AT,BT.T,overwrite_a=1,overwrite_b=1)
            
        #    if (max(np.isnan(T_new)) == 1):     #Check relation
        #        flag = 0
        #        Tobs = 0
        #        break
        
        T_new = T_new[0:tT+1,:]
        
        
        
        Tobs[idinj,:,:] = np.vstack([T_new[:,int(Prod_id[0])],T_new[:,int(Prod_id[1])],T_new[:,int(Prod_id[2])]]).T
    #End of tomography loop
    return Tobs
    
    
def DFN_transport_sensitivity(DFN,SR_coord,limit):  
    #Modified transport model, to identify transportive fractures
    #inputs: DFN + Source-Receiver locations(SR_coord)
    #Original by dr. Mohammadreza Jalali (ETH Zurich)
    
    DFN_Coord_File = DFN[DFN[:,4]==1,:]
    SR_Coord_File = SR_coord
    
    theta1 = np.radians(-12)
    theta2 = np.radians(58.9)
    theta = [0,theta1,theta2]
    
    
    #Input parameters
    gs = 1          # Limit size to divide up larger fractures (discretization length)
    eps = 0.1       # Maximum allowed error for the connectivity matrix
    Visc = 0.001    # Fluis viscosity [Pa.s]
    cf = 1.0e-9     # Fluid compressibility [1/Pa]
    Aper_1 = 0.0015 # Fracture aperture [m]
    Aper_2 = 0.001
    Dt_P = 0.5      # Pressure time step for steady state [sec]
    DP_time = 2500
    n_time_P = int(DP_time/Dt_P) # Maximum number of pressure time steps
    P_Inj = 3e5     # Injection pressure [Pa]
    P_Prod = 1e5    # Production pressure [Pa]
    P_ini = 2e5     # Initial pressure in the fractures [Pa]
    Dt_T = 20       # Temperature time step [sec]
    DT_time = 2000  # Cooling/Warming time [sec]
    K_T = 0.6       # Thermal conductivity of water @ 20 C [J/m.s.C]
    Rho_f = 1000    # Density of water [kg/m3]
    Cf = 420        # Specific heat of water [J/kg.C]
    T_ini = 20      # Initial temperature of fractures [C]
    T_Inj = 40      # Injection temperature
    epsl = 10 # Convergence criteria ???
    Kapa = K_T/(Rho_f*Cf)
    
    # Create the New Fracture Coordinate and Connectivity Matrix
    # DFN_mat = [0-set_id ,1-xc, 2-yc, 3-length, 4-angle, $$$5-set_id$$$, 5-xbeg, 6-ybeg, 7-xend, 8-yend]
    # Connect_Mat: [i1 i2 i3 i4 i5 i6]
    
    Frac_Seg_Mat = np.zeros([np.shape(DFN_Coord_File)[0],9])
    for i in range(0,np.shape(DFN_Coord_File)[0]):
        Frac_Seg_Mat[i,:] = [DFN_Coord_File[i,3], DFN_Coord_File[i,0], DFN_Coord_File[i,1], DFN_Coord_File[i,2], theta[int(DFN_Coord_File[i,3])],
                     DFN_Coord_File[i,0] - DFN_Coord_File[i,2]/2*np.cos(theta[int(DFN_Coord_File[i,3])]), DFN_Coord_File[i,1] - DFN_Coord_File[i,2]/2*np.sin(theta[int(DFN_Coord_File[i,3])]),
                     DFN_Coord_File[i,0] + DFN_Coord_File[i,2]/2*np.cos(theta[int(DFN_Coord_File[i,3])]), DFN_Coord_File[i,1] + DFN_Coord_File[i,2]/2*np.sin(theta[int(DFN_Coord_File[i,3])])]
        
    # Segmentation between intersections
    Segments = np.array([[],[],[],[],[],[],[]]).T
    coord_seg_start = Frac_Seg_Mat[:,5:7].T
    coord_seg_end = Frac_Seg_Mat[:,7:9].T
    
    for i in range(0,np.shape(Frac_Seg_Mat)[0]):
        # Rotate all fractures so that the processed fracture is horizontal and centered at (0,0)
        seg_orient = Frac_Seg_Mat[i,4]
        Rot_Mat = np.array([[np.cos(seg_orient), np.sin(seg_orient)],[-np.sin(seg_orient), np.cos(seg_orient)]]) 
        #Coord of the start point of rotated fractures:
        ### coord_seg_start_rot = np.dot((coord_seg_start - np.matlib.repmat(Frac_Seg_Mat[i,1:3],np.shape(coord_seg_start[:,0])[0],1)),Rot_Mat)
        coord_seg_start_rot = np.dot(Rot_Mat,(coord_seg_start - np.matlib.repmat(Frac_Seg_Mat[i,1:3],np.shape(coord_seg_start)[1],1).T))    #some rounding error appear here!!
        #Coord of the end point of rotated fractures:
        coord_seg_end_rot = np.dot(Rot_Mat,(coord_seg_end - np.matlib.repmat(Frac_Seg_Mat[i,1:3],np.shape(coord_seg_end)[1],1).T))
        
        # Index of fractures intersecting the y=0 line (in rotated axes)
        idx_inter = np.where((coord_seg_start_rot[1,:]*coord_seg_end_rot[1,:]) <= 0)
        # Remove the currently processed fracture from the list:
        idx_inter = np.setdiff1d(idx_inter,i)
        
        # X-coordinate of intersection of going-through fracture with the y=0 line
        x_inter = coord_seg_end_rot[1,idx_inter]/(coord_seg_end_rot[1,idx_inter] - coord_seg_start_rot[1,idx_inter])*(
                coord_seg_start_rot[0,idx_inter] - coord_seg_end_rot[0,idx_inter]) + coord_seg_end_rot[0,idx_inter]
        
        # Indices of intersections with the fracture range (actual intersection with the fracture)
        idx_x_int = np.intersect1d(np.where(x_inter>-Frac_Seg_Mat[i,3]/2), np.where(x_inter<Frac_Seg_Mat[i,3]/2))
        
        #Points along fractures (including ends and intersections in rotated frame)
        if (np.size(idx_x_int) !=0):
            point_seg_rot = np.hstack([-Frac_Seg_Mat[i,3]/2,np.sort(x_inter[idx_x_int]),Frac_Seg_Mat[i,3]/2])
        else:
            point_seg_rot = [-Frac_Seg_Mat[i,3]/2,Frac_Seg_Mat[i,3]/2]
        
        #Rotate back
        Rot_Mat = np.array([[np.cos(seg_orient), -np.sin(seg_orient)],[np.sin(seg_orient), np.cos(seg_orient)]]) 
        #Points along joints (including ends and intersections) unrotated
        point_seg = np.dot(Rot_Mat,np.vstack([point_seg_rot,np.zeros(np.size(point_seg_rot))])).T + np.matlib.repmat(Frac_Seg_Mat[i,1:3].T,np.size(point_seg_rot),1)
        
        # Segments matrix
        # 0-sx1,1-sy1,2-sx2, 3-sy2, 4-slength, 5-sangle, 6-set
        for j in range(0,np.size(point_seg[:,0])-1):
            newline = np.array([point_seg[j,0], point_seg[j,1], point_seg[j+1,0], point_seg[j+1,1], point_seg_rot[j+1]-point_seg_rot[j], Frac_Seg_Mat[i,4], Frac_Seg_Mat[i,0]])
            Segments = np.vstack([Segments,newline])
            
    #Miniplot
    #for i in range(0,np.shape(Segments[:,1])[0]):
    #    plt.plot([Segments[i,0],Segments[i,2]],[Segments[i,1],Segments[i,3]])
    
            
    # Redistribute data
    # 0-nset, 1-xc, 2-yc, 3-gridsize, 4-angle, 5-xb, 6-yb, 7-xe, 8-ye
    DFN_Mat = np.zeros([np.shape(Segments)[0],9])
    DFN_Mat[:,0] = Segments[:,6]
    DFN_Mat[:,1] = (Segments[:,0]+Segments[:,2])/2
    DFN_Mat[:,2] = (Segments[:,1]+Segments[:,3])/2
    DFN_Mat[:,3] = Segments[:,4]
    DFN_Mat[:,4] = Segments[:,5]
    DFN_Mat[:,5] = Segments[:,0]
    DFN_Mat[:,6] = Segments[:,1]
    DFN_Mat[:,7] = Segments[:,2]
    DFN_Mat[:,8] = Segments[:,3]
    
    
    
      
    #Subdivide fracture segments
    DFN_Mat2 = np.array([[],[],[],[],[],[],[],[],[]]).T
    for i in range(0,np.shape(DFN_Mat)[0]):
        if DFN_Mat[i,3] <=gs:
            DFN_Mat2 = np.vstack([DFN_Mat2,DFN_Mat[i,:]])
        else:
            n_seg = np.floor(DFN_Mat[i,3]/gs)
            d_seg = DFN_Mat[i,3]/n_seg
            for j in range(0,int(n_seg)):
                newline0 = DFN_Mat[i,0]
                newline1 = DFN_Mat[i,5]+(j-0.5)*d_seg*np.cos(DFN_Mat[i,4])
                newline2 = DFN_Mat[i,6]+(j-0.5)*d_seg*np.sin(DFN_Mat[i,4])
                newline3 = d_seg
                newline4 = DFN_Mat[i,4]
                newline5 = DFN_Mat[i,5]+j*d_seg*np.cos(DFN_Mat[i,4])
                newline6 = DFN_Mat[i,6]+j*d_seg*np.sin(DFN_Mat[i,4])
                newline7 = DFN_Mat[i,5]+(j+1)*d_seg*np.cos(DFN_Mat[i,4])
                newline8 = DFN_Mat[i,6]+(j+1)*d_seg*np.sin(DFN_Mat[i,4])
                DFN_Mat2 = np.vstack([DFN_Mat2,[newline0,newline1,newline2,newline3,newline4,newline5,newline6,newline7,newline8]])
                
    DFN_Mat = DFN_Mat2
    
    #DFN_Mat = np.delete(DFN_Mat,DFN_Mat[:,3]<eps,axis =0) # Remove very small fracture segments (artifacts)
    
    #Miniplot2
    #for i in range(0,np.shape(DFN_Mat[:,1])[0]):
    #    plt.plot([DFN_Mat[i,5],DFN_Mat[i,7]],[DFN_Mat[i,6],DFN_Mat[i,8]])
    
    #Connectivity matrix
    Connect_Mat = np.ones([np.shape(DFN_Mat[:,1])[0],6])*(-999)   #fixed in v1.1

    
    for i in range(0,np.size(DFN_Mat[:,1])):
        k = 0
        for j in range(0,np.size(DFN_Mat[:,1])):
            if (i!=j):
                if ((abs(DFN_Mat[i,5]-DFN_Mat[j,5])<=eps and abs(DFN_Mat[i,6]-DFN_Mat[j,6])<=eps) or
                    (abs(DFN_Mat[i,7]-DFN_Mat[j,7])<=eps and abs(DFN_Mat[i,8]-DFN_Mat[j,8])<=eps) or
                    (abs(DFN_Mat[i,5]-DFN_Mat[j,7])<=eps and abs(DFN_Mat[i,6]-DFN_Mat[j,8])<=eps) or
                    (abs(DFN_Mat[i,7]-DFN_Mat[j,5])<=eps and abs(DFN_Mat[i,8]-DFN_Mat[j,6])<=eps)):
                    Connect_Mat[i,k] = j
                    k = k+1
                
    #Debug DFN_Mat using the connectivity matrix
    badsector = np.array(np.where(Connect_Mat[:,0]==-999)).T
    DFN_Mat = np.delete(DFN_Mat,badsector, axis=0)
    #Correct Con matrix again
    for i in range(0,np.size(badsector)):
        Connect_Mat = np.delete(Connect_Mat,badsector[i], axis=0)
        Connect_Mat = np.delete(Connect_Mat,np.where(Connect_Mat == badsector[i]),axis = 0)
        Connect_Mat[np.where(Connect_Mat > badsector[i])] = Connect_Mat[np.where(Connect_Mat > badsector[i])] - np.ones(np.shape(Connect_Mat[np.where(Connect_Mat > badsector[i])]))
        badsector = badsector - np.ones([np.shape(badsector)[0],1])
    
    
    # Select Source-Receiver points
    ninj = 0
    nprod = 0
    Inj_id = np.zeros(3)
    Prod_id = np.zeros(3)
    DFN_coords = DFN_Mat[:,1:3]
    for i in range(0,np.shape(SR_Coord_File)[0]):
        id1 = scipy.spatial.cKDTree(DFN_coords).query(SR_Coord_File[i,1:3])[1]   #Select closest point in DFN to the SR point
        if (SR_Coord_File[i,0]==1):
            Inj_id[ninj] = id1
            ninj = ninj + 1
        elif (SR_Coord_File[i,0]==-1):
            Prod_id[nprod] = id1
            nprod = nprod + 1
            
    # Estimate the Coefficient of the Pressure Diffusion PDE
    # Create Conductivity & Storativity Matrix
    num_Frac = np.shape(DFN_Mat)[0]
    AP = np.zeros([num_Frac,num_Frac])  # Init pressure conductivity Matrix(LHS)
    BP = np.zeros(num_Frac) # Init pressure storativity Vector (RHS)
    
    Gamma_P = np.zeros(num_Frac)
    Gamma_P_tot = np.zeros(num_Frac)
    Etha_P = np.zeros(num_Frac)
    
    for i in range(0,num_Frac):
        if (DFN_Mat[i,0] == 1):
            Gamma_P[i] = Dt_P*Aper_1**3/(6*Visc*DFN_Mat[i,3])    # Transmissivity 1
            Etha_P[i] = Aper_1*cf*DFN_Mat[i,3]                  # Storativity 1
        else:
            Gamma_P[i] = Dt_P*Aper_2**3/(6*Visc*DFN_Mat[i,3])    # Transmissivity 2
            Etha_P[i] = Aper_2*cf*DFN_Mat[i,3]                  # Storativity 2
            
    for i in range(0,num_Frac):
        Gamma_P_tot[i] = Gamma_P[i]
        for j in range(0,6):
            cn = int(Connect_Mat[i,j])
            if (cn >= 0):
                Gamma_P_tot[i] = Gamma_P_tot[i] + Gamma_P[cn]   # if and g1g2 based on index
                AP[i,cn] = Gamma_P[i]*Gamma_P[cn]
                AP[i,i] = AP[i,i]-Gamma_P[i]*Gamma_P[cn]
        AP[i,i] = AP[i,i]/Gamma_P_tot[i] - Etha_P[i]
        for j in range(0,6):
            cn = int(Connect_Mat[i,j])
            if (cn >= 0):
                AP[i,cn] = AP[i,cn]/Gamma_P_tot[i]
                
    #for idinj in range(0,3) # Loop for injection location
    Tobs = np.zeros([3,int(DT_time/Dt_T)+1,3])
    DFN_active0 = np.zeros([3,num_Frac,5])
    DFN_active = np.zeros([3,num_Frac,5])
    for idinj in range(0,3):
    
        #Solve for steady-state pressure
        tp = -1
        #timeP = np.linspace(Dt_P,DP_time,n_time_P)
        P_old = np.ones(num_Frac)
        P_new = np.zeros([n_time_P,num_Frac])  
        BP = np.zeros(num_Frac)
        APl = np.copy(AP)
        
        
        # Solve Finite Difference pressure
        while (tp < n_time_P-1):
            tp = tp + 1
            
            #Apply initial conditions
            if (tp==0):
                P_old = P_old*P_ini     # Initial pressure
            else:
                P_old[0:num_Frac] = P_new[tp-1,:]
                
            for i in range(0,num_Frac):
                BP[i] = -Etha_P[i]*P_old[i]
                
            # Apply boundary conditions
            # Injection points
            APl[int(Inj_id[idinj]),:] = 0
            APl[int(Inj_id[idinj]),int(Inj_id[idinj])] = 1
            BP[int(Inj_id[idinj])] = P_Inj
            #Production points
            for i in range(0,np.shape(Prod_id)[0]):
                APl[int(Prod_id[i]),:] = 0
                APl[int(Prod_id[i]),int(Prod_id[i])] = 1
                BP[int(Prod_id[i])] = P_Prod
                
            # Solver for pressure
            #AP2 =  #SPARSE MATRIX NEEDED for performance
            AP2 = scipy.sparse.csc_matrix(APl)
            
            #P_new[tp,:] = np.linalg.lstsq(APl,BP.T)[0]
            P_new[tp,:] = scipy.sparse.linalg.spsolve(AP2,BP.T)
            error = 0
            for i in range(0,num_Frac):
                if (abs(P_new[tp,i]-P_old[i]) > error):
                    error = abs(P_new[tp,i] - P_old[i])     #EZT LE KELL ELLENORIZNI + EGYSZERUSITENI!!!
            if error < epsl:
                break
        tss = tp
        
        P_new = P_new[0:tss+1,:]
        
        #Estimate steady-state velocity
        num_up = np.zeros(num_Frac)
        V = np.zeros(num_Frac)
        for i in range(0,num_Frac):
            P_Max = P_new[tss,i]
            num_up[i] = i
            for j in range(0,6):
                cn = int(Connect_Mat[i,j])
                if (cn >= 0 and P_new[tp,cn] > P_Max):
                    P_Max = P_new[tp,cn]
                    num_up[i] = cn
            
            if (DFN_Mat[i,0] == 1):
                V[i] = Aper_1**2/(12*Visc*DFN_Mat[i,3])*(P_Max-P_new[tp,i])
            else:
                V[i] = Aper_2**2/(12*Visc*DFN_Mat[i,3])*(P_Max-P_new[tp,i])
                
        #Plot velocity
        
             
        # Estimate the Coefficient of the Temperature Diffusion PDE
        # Create Conductivity & Storativity Matrix
        Gamma_T = np.zeros(num_Frac)
        XI_T = np.zeros(num_Frac)
        Etha_T = np.zeros(num_Frac)
        for i in range(0,num_Frac):
            if (DFN_Mat[i,0] == 1):
                Gamma_T[i] = 2*Dt_T*Kapa*Aper_1/DFN_Mat[i,3]
                XI_T[i] = Aper_1*Dt_T*V[i]
                Etha_T[i] = Aper_1*DFN_Mat[i,3]
            else:
                Gamma_T[i] = 2*Dt_T*Kapa*Aper_2/DFN_Mat[i,3]
                XI_T[i] = Aper_2*Dt_T*V[i]
                Etha_T[i] = Aper_2*DFN_Mat[i,3]
                
        #Initialize temperature matrices
        AT = np.zeros([num_Frac,num_Frac])    #Conductivity matrix(LHS)
        BT = np.zeros(num_Frac)     #Storativity vector(RHS)
        Gamma_T_tot = np.zeros(num_Frac)
        for i in range(0,num_Frac):
            Gamma_T_tot[i] = Gamma_T[i]
            for j in range(0,5):
                cn = int(Connect_Mat[i,j])
                if (cn >=0):
                    Gamma_T_tot[i] = Gamma_T_tot[i]+Gamma_T[cn]
                    AT[i,cn] = Gamma_T[i]*Gamma_T[cn]
                    AT[i,i] = AT[i,i]-(Gamma_T[i]*Gamma_T[cn])
            AT[i,i] = AT[i,i]/Gamma_T_tot[i]-Etha_T[i]
            for j in range(0,5):
                cn = int(Connect_Mat[i,j])
                if (cn >=0):
                    AT[i,cn] = AT[i,cn]/Gamma_T_tot[i]
            
            AT[i,i] = AT[i,i]-XI_T[i]
            AT[i,int(num_up[i])] = AT[i,int(num_up[i])]+XI_T[i]
            
        #Solve for temperature
        tT = -1
        n_time_T = int(DT_time/Dt_T)
        #time = np.linspace(Dt_T,DT_time,n_time_T)
        T_new = np.zeros([n_time_T+1,num_Frac]) 
        
        #Solve Finite Difference Temperature
        while (tT < n_time_T):
            tT = tT + 1
            if (tT==0):
                T_old = np.ones(num_Frac)*T_ini
            else:
                T_old = T_new[tT-1,:]
                
            for i in range(0,num_Frac):
                BT[i] = -Etha_T[i]*T_old[i]
                
            #Apply boundary conditions
            AT[int(Inj_id[idinj]),:] = 0
            AT[int(Inj_id[idinj]),int(Inj_id[idinj])] = 1
            BT[int(Inj_id[idinj])] = T_Inj
            
            #Solver
            AT2 = scipy.sparse.csc_matrix(AT)
            T_new[tT,:] = scipy.sparse.linalg.spsolve(AT2,BT.T)
        #    if (max(np.isnan(T_new)) == 1):     #Check relation
        #        flag = 0
        #        Tobs = 0
        #        break
        
        T_new = T_new[0:tT+1,:]
        
        
        
        Tobs[idinj,:,:] = np.vstack([T_new[:,int(Prod_id[0])],T_new[:,int(Prod_id[1])],T_new[:,int(Prod_id[2])]]).T
        active_id = np.where(abs(T_new[tT,:]-T_ini) > limit )
        #active_id = np.where(V > 0 )
        DFN_active0[idinj,0:np.shape(DFN_Mat[active_id,:])[1],:] = DFN_Mat[active_id,0:5]
        col0 = DFN_active0[idinj,:,1]
        col1 = DFN_active0[idinj,:,2]
        col2 = DFN_active0[idinj,:,3]
        col3 = DFN_active0[idinj,:,0]
        col4 = np.ones(np.shape(DFN_active0)[1])
        DFN_active[idinj,:,:] = np.vstack([col0,col1,col2,col3,col4]).T
        #DFN_active = np.hstack([DFN_active0[:,1:5],DFN_active0[:,0]])
    #End of tomography loop
    return Tobs,DFN_active