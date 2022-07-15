    #Input parameters
    #Discretization
    gs = 1          # Limit size to divide up larger fractures (discretization length)
    dx = 1
    eps = 0.1       # Maximum allowed error for the connectivity matrix
    epsl = 10 # Convergence criteria ???
    
    #DFN parameters
    Visc = 0.001    # Fluis viscosity [Pa.s]
    cf = 1.0e-9     # Fluid compressibility [1/Pa]
    Aper_1 = 0.0015 # Fracture aperture [m]
    Aper_2 = 0.001
    Rho_f = 1000    # Density of water [kg/m3]
    Cf = 420        # Specific heat of water [J/kg.C]
    K_T = 0.6       # Thermal conductivity of water @ 20 C [J/m.s.C]
    
    #Experiment parameters
    Dt_P = 0.5      # Pressure time step for steady state [sec]
    DP_time = 2500
    n_time_P = int(DP_time/Dt_P) # Maximum number of pressure time steps
    P_Inj = 3e5     # Injection pressure [Pa]
    P_Prod = 1e5    # Production pressure [Pa]
    P_ini = 2e5     # Initial pressure in the fractures [Pa]
    
    Dt_T = 20       # Temperature time step [sec]
    DT_time = 2000  # Cooling/Warming time [sec]
    T_ini = 20      # Initial temperature of fractures [C]
    T_Inj = 40      # Injection temperature
    
    Kapa = K_T/(Rho_f*Cf)