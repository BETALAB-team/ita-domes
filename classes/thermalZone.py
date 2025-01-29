'''IMPORTING MODULES'''

import sys
import statistics
import os
from copy import deepcopy as cp
import pvlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from classes.Geometry import  Surface, SurfaceInternalMass, SurfaceInternalAdjacent
from classes.Envelope import loadEnvelopes
from classes.EndUse import loadArchetype
from classes.AHU import AirHandlingUnit
from classes.PVPlant import DistrictPVGIS
from classes.BuildingsPlants import Plants
#from mpl_toolkits.mplot3d import Axes3D

#%% Useful functions

def impedenceParallel(R,C,T_RA = 5):
    '''
    equivComplexRes Given two vectors (thermal resistances r and thermal 
    capacitances c) of length m (number of walls of the same type, ie either 
    IW or AW), calculates the equivalent complex thermal resistance Zeq
    according to T_RA (period in days)
    '''    
    
    #T_RA = 5;       % Period = 5 days
    omega_RA = 2*np.pi/(86400*T_RA)
    z = np.zeros(len(R),complex)
    z = (R+1j/(omega_RA*C))   # vettore delle Z 
    
    Z1eq = 1/(sum(1/z))                # Z equivalente
    
    R1eq = np.real(Z1eq)
    C1eq = 1/(omega_RA*np.imag(Z1eq))
    return R1eq, C1eq


def tri2star(T1,T2,T3):
    '''
    #tri2star Transforms three resistances in triangular connection into
    #three resistances in star connection
    '''
    
    T_sum = T1+T2+T3
    S1 = T2*T3/T_sum
    S2 = T1*T3/T_sum
    S3 = T2*T1/T_sum
    return S1, S2, S3


def longWaveRadiation(theta_a,SSW = 1):
    '''
    Estimation of sky and ground temperatures via vdi6007 model:
        theta_a outdoor air temperature [°C]
        SSW factor to count the clear non-clear sky
    '''

    Ea_1 = 9.9*5.671*10**(-14)*(273.15+theta_a)**6
    
    alpha_L = 2.30 - 7.37*10**(-3)*(273.15+theta_a)
    alpha_M = 2.48 - 8.23*10**(-3)*(273.15+theta_a)
    alpha_H = 2.89 - 1.00*10**(-2)*(273.15+theta_a)
    
    Ea = Ea_1*(1+(alpha_L+(1-(1-SSW)/3)*alpha_M+((1-(1-SSW)/3)**2)*alpha_H)*((1-SSW)/3)**2.5)
    Ee = -(0.93*5.671*10**(-8)*(273.15+theta_a)**4+(1-0.93)*Ea)
    
    theta_erd = ((-Ee/(0.93*5.67))**0.25)*100 - 273.15  # [°C]
    theta_atm = ((Ea/(0.93*5.67))**0.25)*100 - 273.15   # [°C]  

    return Ea, Ee, theta_erd, theta_atm


def loadHK(perc_rad, perc_rad_aw, perc_altro_irr, A_aw, A_raum):
    '''
    loadHK  -  Distribution of heat load on the nodes (surface nodes and air node) based on heat emitters
    of the building
    
    INPUT 
    perc_rad (sigma_fhk): percentage of heat flow by radiant floors
    perc_rad_aw (sigma_fhk_aw): percentage of radiant floors installed on AW walls 
    perc_altro_irr (sigma_rad_str): percentage of radiant load (out of total which is
    rad+conv) by other types of heat emitters (e.g.: fan-coils, radiators)
    A_aw = sum of the exterior opaque building components
    A_raum = sum of internal partitions (all IW components) and exterior
    opaque building components
    
    OUTPUT
    sigma_hk_iw: radiant heat load  on IW building components (on surface
    node IW)
    sigma_hk_aw: radiant heat load  on AW building components (on surface
    node AW)
    sigma_hk_kon: convective heat load (on air node)
    '''

# %Note: the sum of the 3 outputs must be equal to 1
    
    if perc_rad == 1:

        perc_rad_iw = 1 - perc_rad_aw
        sigma_hk_iw = perc_rad_iw
        sigma_hk_aw = perc_rad_aw
        sigma_hk_kon = 0

    elif perc_rad == 0:

        sigma_hk_iw = 0
        sigma_hk_aw = 0
        sigma_hk_kon = 1
    
    else:
        perc_rad_iw = 1 - perc_rad_aw
        perc_altro = 1 - perc_rad
        perc_altro_irr = perc_altro*perc_altro_irr
        sigma_hk_iw = perc_altro_irr*(A_raum-A_aw)/A_raum + perc_rad_iw
        sigma_hk_aw = perc_altro_irr*A_aw/A_raum + perc_rad_aw
        sigma_hk_kon = 1 - sigma_hk_iw - sigma_hk_aw
        
    return [sigma_hk_iw, sigma_hk_aw, sigma_hk_kon]

#%%

class ThermalZone:
    '''
    thermal zone class
    
    __init__:
        zone number
        name of the building
        envelope object of the zone
        schedule archetype object of the zone
        list of the surfaces
        volume
        zone area
        
    zoneParameter13790 calculates 1C params. No input
    
    calculate_zone_loads_ISO13790 calculates 1C zone loads:
        Solar Gains vector [W/m2]
        average difference between t air and t sky [°C]
        .
        .       
        
    Sensible1C 1C system solver:
        solver typology phi set or T set
        vector with ventilation and infiltration heat transfer coefficients [W/K]
        Setpoint temperature [°C]
        supply setpoint temperature [°C]
        external temperature [°C]
        time step duration [s]
        plant power setpoint [W]
        
    zoneParameter6007 calculates 2C params. No input
    
    calculate_zone_loads_vdi6007 calculate 2C zone loads:
        external temeprature [°C]
        Solar Gains dataframe [W/m2]
        
    Sensible2C 2C system solver:
        solver typology phi set or T set
        gains vector [W]
        vector with ventilation and infiltration heat transfer coefficients [W/K]
        external and sun-air temperature [°C]
        supply setpoint temperature [°C]
        time step [s]
        Setpoint temperature [°C]
        plant power setpoint [W]

    solveTZ thermal zone balances solver:
        time step [-]
        external temperature [°C]
        external relative humidity [-]
        external saturation pressure [Pa]
        time step duration [s]
        mode 2C or 1C
    '''
    
    
    '''
    Class attributes
    '''
    htr_ms = 9.1
    his = 3.45
    rho_air = 1.2                                                              # [kg/m3]
    cp_air = 1000.                                                              # [J/(kg K)]
    sens_frac = 0.5462                                                         # Based on Standard ISO 18523
    p_atm = 101325.                                                             # [Pa]
    r_0 = 2501000.                                                              # [J/kg]
    cpv = 1875.                                                                 # [J/(kg K)]
    x_m0 = 0.0105                                                              # [kg_v/kg_as]
    x_m0DD = 0.0105                                                            # [kg_v/kg_as]
    T_wall_0 =  15   
    
    def __init__(self,zoneNumber,buildingName,envelope,sched_db,surfList,volume,zone_area,l):
        self.name ='Zone ' + str(zoneNumber)
        self.schedules = sched_db
        
        
        self.Strat={'ExtWall':envelope.ExtWall,\
        'IntWall':envelope.IntWall,\
        'IntCeiling':envelope.IntCeiling,\
        'GroundFloor':envelope.GroundFloor,\
        'Roof':envelope.Roof,
        'Window':envelope.Window,
        'IntFloor':envelope.IntFloor}                            
        self.V = volume
        self.Ca = self.V*self.rho_air*self.cp_air 
        self.surfaces = {}
        self.zone_area = zone_area
        self.Ta0 = 15
        self.theta_m0 = 15
        self.theta_m0_vdi = [15, 15]                                           # aw the first, iw the second

        i = 0
        self.Araum = 0
        self.Aaw = 0
        self.Tot_glazed_area = 0
        self.Tot_opaque_area = 0
        for surface in surfList:
            i += 1
            self.surfaces[('Surface '+str(i))]=surface
            self.Araum += self.surfaces[('Surface '+str(i))].area
            if surface.type == 'ExtWall' or surface.type == 'GroundFloor' or surface.type == 'Roof' :
                self.Aaw += self.surfaces[('Surface '+str(i))].area
            if surface.type == 'ExtWall':
                self.Tot_glazed_area += surface.glazedArea
                self.Tot_opaque_area += surface.opaqueArea
                
        '''Vectors inizialization'''
        self.heatFlow = np.zeros(l)
        self.Air_temp = np.zeros(l)
        self.theta_m0_2c = np.zeros([l,2])
        self.ext_surf_heat_gain = np.zeros(l)                                                                     
        self.RH_i = np.zeros(l)
        self.latentFlow = np.zeros(l)
        self.x_ext = np.zeros(l)
        self.G_v = np.zeros(l)
        self.G_da_inf = np.zeros(l)
        self.G_da_vent = np.zeros(l)
        self.x_int = np.zeros(l)
        self.T_sup = np.zeros(l)
        self.ZoneAHU = AirHandlingUnit(l)
        self.T_wall_0_vector = np.zeros(l)
        self.R_lim_ext_wall_2C = self.Strat['ExtWall'].R_se / (self.Tot_opaque_area + self.Tot_glazed_area)
        self.H_lim_ext_wall_1C = self.Tot_opaque_area / self.Strat['ExtWall'].R_se
        
        self.Htr_is = 0
        self.Htr_w = 0
        self.Htr_ms = 0
        self.Htr_em = 0
        self.Cm = 0
        self.DenAm = 0
        self.Atot = 0
        self.Htr_op = 0
        
        self.RrestAW = 0
        self.R1AW = 0
        self.RalphaStarAW = 0
        self.RalphaStarIL = 0
        self.RalphaStarIW = 0
        self.R1IW = 0
        self.C1AW = 0
        self.C1IW = 0               
        
    def zoneParameter13790(self):
        
        self.Htr_int = []
        
        for surface in self.surfaces.values():
            if surface.type == 'ExtWall' or surface.type == 'IntWall' or surface.type == 'IntCeiling' or surface.type == 'Roof' or surface.type == 'GroundFloor':
                self.Cm += surface.opaqueArea*(self.Strat[surface.type].k_int)
                self.DenAm += surface.opaqueArea*(self.Strat[surface.type].k_int)**2
            if surface.type == 'IntFloor':
                self.Cm += surface.opaqueArea*(self.Strat[surface.type].k_est)
                self.DenAm += surface.opaqueArea*(self.Strat[surface.type].k_est)**2
            self.Atot += surface.area
            
            if surface.type == 'ExtWall' or surface.type == 'GroundFloor' or surface.type == 'Roof' :
                self.Htr_op += surface.opaqueArea*(self.Strat[surface.type].U)
                self.Htr_w += surface.glazedArea*(self.Strat['Window'].U)
                
            if type(surface) == SurfaceInternalAdjacent:
                self.Htr_int.append({'H':surface.opaqueArea*self.Strat[surface.type].U,'AdjacentZone':surface.adjacentZone})
        
        self.Am = self.Cm**2/self.DenAm
        self.Htr_ms = self.Am*self.htr_ms
        self.Htr_em = 1/(1/self.Htr_op - 1/self.Htr_ms)
        self.Htr_is = self.his*self.Atot
        self.UA_tot = self.Htr_op + self.Htr_w

        
    def calculate_zone_loads_ISO13790(self,Solar_gain,dT_er,eps_se,h_r):
        
        self.phi_int = (self.schedules.people*self.sens_frac+self.schedules.appliances+self.schedules.lighting)
        phi_sol_gl_tot = 0
        phi_sol_op_tot = 0

        for i in self.surfaces.values():
            
            phi_sol_op = 0
            phi_sol_gl = 0
            
            if i.type == 'ExtWall' or i.type == 'Roof':
                irradiance = Solar_gain[str(float(i.azimuth_round))][str(float(i.height_round))]  
                BRV = irradiance['direct'].to_numpy()                           
                TRV = irradiance['global'].to_numpy()
                DRV = TRV -BRV
                A_ww = i.glazedArea
                self.A_ew = i.opaqueArea
                
                ''' Glazed surfaces'''
                F_sh = self.Strat['Window'].F_sh
                F_w = self.Strat['Window'].F_w
                F_f = self.Strat['Window'].F_f
                F_so = self.Strat['Window'].F_so
                AOI = irradiance['AOI'].to_numpy()
                shgc = interpolate.splev(AOI, self.Strat['Window'].SHGC_profile, der=0)
                shgc_diffuse = interpolate.splev(70, self.Strat['Window'].SHGC_profile, der=0)
                if i.OnOff_shading == 'On':
                    phi_sol_gl = F_so*(BRV*F_sh*F_w*(1-F_f)*shgc*A_ww*i.shading_effect)+F_so*DRV*F_sh*F_w*(1-F_f)*shgc_diffuse*A_ww
                else:
                    phi_sol_gl = F_so*(BRV*F_sh*F_w*(1-F_f)*shgc*A_ww)+F_so*DRV*F_sh*F_w*(1-F_f)*shgc_diffuse*A_ww
            
                '''Opaque surfaces'''
                if i.type == 'ExtWall':
                    self.F_so_op = F_so
                    self.F_r = i.F_r
                    self.alpha = self.Strat['ExtWall'].alpha_est
                    self.sr_ew = self.Strat['ExtWall'].R_se
                    self.U_ew_net = self.Strat['ExtWall'].U_net
                    if i.OnOff_shading == 'On':
                        phi_sol_op = self.F_so_op*(BRV*i.shading_effect + DRV)*self.alpha*self.sr_ew*self.U_ew_net*self.A_ew-self.F_r*self.sr_ew*self.U_ew_net*self.A_ew*h_r*dT_er
                    else:
                        phi_sol_op = self.F_so_op*TRV*self.alpha*self.sr_ew*self.U_ew_net*self.A_ew-self.F_r*self.sr_ew*self.U_ew_net*self.A_ew*h_r*dT_er
                    
                else:
                    TRH = Solar_gain['0.0']['0.0']['global'].to_numpy()
                    self.F_r = i.F_r
                    self.alpha = self.Strat['Roof'].alpha_est
                    self.sr_rf = self.Strat['Roof'].R_se
                    self.U_rf_net = self.Strat['Roof'].U_net
                    phi_sol_op = TRH*self.alpha*self.sr_rf*self.U_rf_net*self.A_ew-self.F_r*self.sr_rf*self.U_rf_net*self.A_ew*h_r*dT_er
                                    
                
            # '''Opaque surfaces'''
            # if i.type == 'ExtWall':
            #     self.F_so_op = F_so
            #     self.F_r = i.F_r
            #     self.alpha = self.Strat['ExtWall'].alpha_est
            #     self.sr_ew = self.Strat['ExtWall'].R_se
            #     self.U_ew_net = self.Strat['ExtWall'].U_net
            #     if i.OnOff_shading == 'On':
            #         phi_sol_op = self.F_so_op*(BRV*i.shading_effect + DRV)*self.alpha*self.sr_ew*self.U_ew_net*self.A_ew-self.F_r*self.sr_ew*self.U_ew_net*self.A_ew*h_r*dT_er
            #     else:
            #         phi_sol_op = self.F_so_op*TRV*self.alpha*self.sr_ew*self.U_ew_net*self.A_ew-self.F_r*self.sr_ew*self.U_ew_net*self.A_ew*h_r*dT_er
                
            # if i.type == 'Roof':
            #     TRH = Solar_gain['0.0']['0.0']['global'].to_numpy()
            #     self.F_r = i.F_r
            #     self.alpha = self.Strat['Roof'].alpha_est
            #     self.sr_rf = self.Strat['Roof'].R_se
            #     self.U_rf_net = self.Strat['Roof'].U_net
            #     phi_sol_op = TRH*self.alpha*self.sr_rf*self.U_rf_net*self.A_ew-self.F_r*self.sr_rf*self.U_rf_net*self.A_ew*h_r*dT_er
                
            ''' Total solar gain'''    
            phi_sol_gl_tot += phi_sol_gl
            phi_sol_op_tot += phi_sol_op
        self.phi_sol = phi_sol_gl_tot+phi_sol_op_tot
       
        ''' Distribute heat gains to temperature nodes'''
        self.phi_ia = 0.5*self.phi_int
        self.phi_st = (1-self.Am/self.Atot-self.Htr_w/(9.1*self.Atot))*(0.5*self.phi_int + self.phi_sol)
        self.phi_m = self.Am/self.Atot*(0.5*self.phi_int + self.phi_sol)
        
        self.phi_int_medio = np.sum(self.phi_int)/8760 # kW
        self.phi_sol_medio = np.sum(self.phi_sol)/8760 # kW
        
        
          
    def Sensible1C(self, flag, Hve, T_set, T_sup_AHU, T_e, phi_load, tau, phi_HC_set = 0):
        
        return {
        
        'Tset':self.Tset,
        'phiset':self.Phiset
            
        }[flag](Hve, T_set, T_sup_AHU, T_e, phi_load, tau, phi_HC_set)
        
    def Tset(self,Hve, T_set, T_sup_AHU, T_e, phi_load, tau, phi_HC_set):
        phi_ia = phi_load[0]
        phi_st = phi_load[1]
        phi_m = phi_load[2]
        Hve_vent = Hve[0]
        Hve_inf = Hve[1]
        Y = np.zeros((3,3))
        q = np.zeros((3))
    
        Y[0,0] = 1
        Y[0,1] = self.Htr_is
        Y[1,1] = -(self.Htr_is+self.Htr_w+self.Htr_ms)
        Y[1,2] = self.Htr_ms
        Y[2,1] = self.Htr_ms
        Y[2,2] = -self.Cm/tau-self.Htr_em-self.Htr_ms
    
        q[0] = Hve_inf*(T_set-T_e)+Hve_vent*(T_set-T_sup_AHU)-phi_ia+self.Htr_is*T_set
        q[1] = -self.Htr_is*T_set-phi_st-self.Htr_w*T_e
        q[2] = -self.Htr_em*T_e-phi_m-self.Cm*self.theta_m0/tau
        x = np.linalg.inv(Y).dot(q)
        return np.insert(x,1,T_set)
    
    def Phiset(self,Hve, T_set, T_sup_AHU, T_e, phi_load, tau, phi_HC_set):
        phi_ia = phi_load[0]
        phi_st = phi_load[1]
        phi_m = phi_load[2]
        Hve_vent = Hve[0]
        Hve_inf = Hve[1]
        Y = np.zeros((3,3))
        q = np.zeros((3))
    
        Y[0,0] = -(self.Htr_is+Hve_inf+Hve_vent)
        Y[0,1] = self.Htr_is
        Y[1,0] = self.Htr_is
        Y[1,1] = -(self.Htr_is+self.Htr_w+self.Htr_ms)
        Y[1,2] = self.Htr_ms
        Y[2,1] = self.Htr_ms
        Y[2,2] = -self.Cm/tau-self.Htr_em-self.Htr_ms
        
        q[0] = -phi_HC_set-Hve_inf*T_e-Hve_vent*T_sup_AHU-phi_ia
        q[1] = -phi_st - self.Htr_w*T_e
        q[2] = -self.Htr_em*T_e-phi_m-self.Cm*self.theta_m0/tau
        y = np.linalg.inv(Y).dot(q)
        return np.insert(y,0,phi_HC_set)
            
    def zoneParameter6007(self):                             
        R1IW_m = np.array([])
        C1IW_m = np.array([])
        R_IW = np.array([])
        R1AW_v = np.array([])
        C1AW_v = np.array([])
        R_AW = np.array([])
        R1_AF_v = np.array([])
        HAW_v = np.array([])
        HAF_v = np.array([])
        alphaKonAW = np.array([])
        alphaKonIW = np.array([])
        alphaKonAF = np.array([])
        alphaStr = 5 #vdi Value
        alphaKonA = 20 #vdi value
        RalphaStrAW = np.array([])
        RalphaStrIW = np.array([])
        RalphaStrAF = np.array([])
        AreaAW =  np.array([])
        AreaAF =  np.array([])
        AreaIW =  np.array([])
        
        for surface in self.surfaces.values():
            if surface.type == 'ExtWall' or surface.type == 'GroundFloor' or surface.type == 'Roof':
                surface_R1, surface_C1 = self.Strat[surface.type].vdi6007surfaceParams(surface.opaqueArea,True)
                #R1AW_v.append(surface_R1)
                C1AW_v = np.append(C1AW_v,[surface_C1],axis = 0)
                #R_AW = np.append(R_AW,[sum(self.Strat[surface.type].r)],axis=0)
                # considering glazing component
                R_AF_v = (self.Strat['Window'].Rl_w/surface.glazedArea) #Eq 26
                #R1_AF_v = np.append(R1_AF_v,[R_AF_v/6],axis=0) #Jacopo utilizes a different formula, but this is what I understood from the standard
                # this part is a little different in Jacopo model,
                # However this part calculates opaque R, glazed R and insert the parallel ass wall R
                R1AW_v = np.append(R1AW_v,[1/(1/surface_R1+1/R_AF_v)],axis=0)
                #R1AW_v = np.append(R1AW_v,[1/(1/surface_R1+6/R_AF_v)],axis=0) ALTERNATIVA NORMA
                
                HAW_v = np.append(HAW_v,self.Strat[surface.type].U*surface.opaqueArea)
                HAF_v = np.append(HAF_v,self.Strat['Window'].U*surface.glazedArea)
                alphaKonAW = np.append(alphaKonAW,[surface.opaqueArea*(1/self.Strat[surface.type].R_si-alphaStr)],axis=0)
                alphaKonAF = np.append(alphaKonAF,[surface.glazedArea*(1/self.Strat['Window'].Ri_w-alphaStr)],axis=0)
                                  
                RalphaStrAW = np.append(RalphaStrAW,[1/(surface.opaqueArea*alphaStr)])
                RalphaStrAF = np.append(RalphaStrAF,[1/(surface.glazedArea*alphaStr)])
                
                AreaAW = np.append(AreaAW,surface.opaqueArea)
                AreaAF = np.append(AreaAF,surface.glazedArea)
                
            elif surface.type == 'IntCeiling' or surface.type == 'IntWall' or  surface.type=='IntFloor':
                surface_R1, surface_C1 = self.Strat[surface.type].vdi6007surfaceParams(surface.opaqueArea,False)
                R1IW_m = np.append(R1IW_m,[surface_R1],axis=0)
                C1IW_m = np.append(C1IW_m,[surface_C1],axis =0)
                R_IW = np.append(R_IW,[sum(self.Strat[surface.type].r)],axis=0)
                alphaKonIW = np.append(alphaKonIW,[surface.opaqueArea*(1/self.Strat[surface.type].R_si-alphaStr)],axis=0)
                
                #if surface.opaqueArea*alphaStr == 0:
                #    print(surface.name)
                RalphaStrIW = np.append(RalphaStrIW,[1/(surface.opaqueArea*alphaStr)])
                
                AreaIW = np.append(AreaIW,surface.area)
            else:
                print('Error.. surface type not found')
        
        self.R1AW, self.C1AW = impedenceParallel(R1AW_v,C1AW_v)  #eq 22
        self.R1IW, self.C1IW = impedenceParallel(R1IW_m,C1IW_m)
        
        self.RgesAW = 1 / (sum(HAW_v)+sum(HAF_v))  #eq 27

        RalphaKonAW = 1/(sum(alphaKonAW)+sum(alphaKonAF))   #scalar
        RalphaKonIW = 1/sum(alphaKonIW)   #scalar 
        
        if sum(AreaAW) <= sum(AreaIW):
            RalphaStrAWIW = 1/(sum(1/RalphaStrAW)+sum(1/RalphaStrAF))  #eq 29
        else:
            RalphaStrAWIW = 1/sum(1/RalphaStrIW)  #eq 31
            
        self.RrestAW = self.RgesAW - self.R1AW - 1/(1/RalphaKonAW + 1/RalphaStrAWIW) #eq 28
        
        RalphaGesAW_A = 1/(alphaKonA*(sum(AreaAF) + sum(AreaAW)))
        
        if self.RgesAW < RalphaGesAW_A:   # this is differentfrom Jacopo model but equal to the standard
            self.RrestAW = RalphaGesAW_A  #eq 28a
            self.R1AW = self.RgesAW - self.RrestAW - 1/(1/RalphaKonAW + 1/RalphaStrAWIW) #eq 28b
            
            if self.R1AW < 10**(-10):
                self.R1AW = 10**(-10)   # Thresold (only numerical to avoid division by zero)  #eq 28c
                
        self.RalphaStarIL, self.RalphaStarAW, self.RalphaStarIW = tri2star(RalphaStrAWIW,RalphaKonIW,RalphaKonAW)
        self.UA_tot = sum(HAW_v)+sum(HAF_v)   
        self.Htr_op = sum(HAW_v)
        self.Htr_w = sum(HAF_v) 

    def calculate_zone_loads_vdi6007(self,T_ext,Solar_gain):
        #rho_ground = 0.2
        #va calcolata
        '''
        Eerd = Solar_gain['0.0']['0.0']['Global']*rho_ground                          #
        Eatm = Solar_gain['0.0']['0.0']['Global']-Solar_gain['0.0']['0.0']['Direct']
        
        T_ext vettore
        
        '''
        
        # Tutta questa parte quì si può mettere in una subrutine diversa (il calcolo delle E e di alpha str)
        Eatm, Eerd, theta_erd, theta_atm = longWaveRadiation(T_ext)
        #fil =  (Eatm + Eerd)*(theta_erd - theta_atm)
        alpha_str_A = 5
        
        theta_eq = np.zeros([len(T_ext),len(self.surfaces)])
        delta_theta_eq_lw = np.zeros([len(T_ext),len(self.surfaces)])
        delta_theta_eq_kw = np.zeros([len(T_ext),len(self.surfaces)])
        theta_eq_w = np.zeros([len(T_ext),len(self.surfaces)])
        frame_factor = 1 - self.Strat['Window'].F_f
        F_sh = self.Strat['Window'].F_sh*self.Strat['Window'].F_so*self.Strat['Window'].F_w
        Q_il_str_A_iw = 0
        Q_il_str_A_aw = 0
        #self.Q_il_str_A = 0 
        
        i = -1
        
        for surface in self.surfaces.values():
            i += 1
            if surface.type == 'ExtWall' or surface.type == 'Roof':
                if surface.OnOff_shading == 'On':
                    shading = surface.shading_effect
                else:
                    shading = 1
                irradiance = Solar_gain[str(float(surface.azimuth_round))][str(float(surface.height_round))]
                AOI = irradiance['AOI'].to_numpy()
                BRV = irradiance['direct'].to_numpy()                           
                TRV = irradiance['global'].to_numpy()
                DRV = TRV -BRV
                phi = surface.F_r  #eventualmente si può importare dalla surface
                alpha_a = self.Strat[surface.type].alpha_conv_est + alpha_str_A
                eps_F = 0.9
                #eps_F = self.Strat[surface.type] ANDREBBE AGGIUNTO NEI DATI DI INPUT
                delta_theta_eq_lw[:,i] = ((theta_erd - T_ext)*(1-phi) + (theta_atm - T_ext)*phi)*(eps_F*alpha_str_A)/(0.93*alpha_a)
                delta_theta_eq_kw[:,i] = (BRV*shading+(TRV-BRV))*self.Strat[surface.type].alpha_est/alpha_a
                theta_eq[:,i] = (T_ext + delta_theta_eq_lw[:,i] + delta_theta_eq_kw[:,i])*self.Strat[surface.type].U*surface.opaqueArea/self.UA_tot
                theta_eq_w[:,i] = (T_ext + delta_theta_eq_lw[:,i])*self.Strat['Window'].U*surface.glazedArea/self.UA_tot
                
                shgc = interpolate.splev(AOI, self.Strat['Window'].SHGC_profile, der=0)
                shgc_diffuse = interpolate.splev(70, self.Strat['Window'].SHGC_profile, der=0)                                                                                                                                                         
                # Jacopo quì usa come A_v l'area finestrata, mentre la norma parla di area finestrata + opaca per la direzione 
                Q_il_str_A_iw += frame_factor*F_sh*surface.glazedArea*(shgc*BRV*shading+shgc_diffuse*(TRV-BRV))*((self.Araum - self.Aaw)/(self.Araum -surface.glazedArea))
                Q_il_str_A_aw += frame_factor*F_sh*surface.glazedArea*(shgc*BRV*shading+shgc_diffuse*(TRV-BRV))*((self.Aaw - surface.glazedArea)/(self.Araum -surface.glazedArea))
                   
            if surface.type == 'GroundFloor':
                theta_eq[:,i] = T_ext*self.Strat[surface.type].U*surface.opaqueArea/self.UA_tot
        
        #self.Q_il_str_A = self.Q_il_str_A.to_numpy()
        #self.carichi_sol = (Q_il_str_A_iw+ Q_il_str_A_aw).to_numpy()
        
        self.theta_eq_tot = theta_eq.sum(axis = 1) + theta_eq_w.sum(axis = 1) 
        
        conv_people = 1
        conv_app = 1
        
        Q_il_str_I = (self.schedules.people*self.sens_frac*(1-conv_people)+self.schedules.appliances*(1-conv_app)+self.schedules.lighting*(1-conv_app))
        self.Q_il_kon_I = (self.schedules.people*self.sens_frac*(conv_people)+self.schedules.appliances*(conv_app)+self.schedules.lighting*(conv_app))
        
        Q_il_str_I_iw = Q_il_str_I * (self.Araum-self.Aaw)/self.Araum
        Q_il_str_I_aw = Q_il_str_I * self.Aaw/self.Araum
        
        self.Q_il_str_iw =  Q_il_str_A_iw + Q_il_str_I_iw
        self.Q_il_str_aw =  Q_il_str_A_aw + Q_il_str_I_aw
        
        
        sigma_fhk = 0         # percentage of heating from from radiant floor 
        sigma_fhk_aw = 0      # percentage of radiant floor systems embedded in external (AW) walls 
        sigma_fhk_iw = 1 - sigma_fhk_aw
        
        sigma_hk_str = 1    # 50% rad, 50% conv
        
        self.sigma = loadHK(sigma_fhk, sigma_fhk_aw, sigma_hk_str, self.Aaw, self.Araum)                                                          
    
    def Sensible2C(self, flag, phi, H_ve, theta_bound, theta_sup, tau, theta_set = 20, Q_hk = 0):
        # function x = buildingLS_2C_Tset(phi, theta_bound, params, anteil, theta_set, theta_m0, tau)
        # %buildingLS_2C_Tset solves the linear system (Y*x = q) of VDI6007 at each
        # %iteration with a given setpoint temperature
        
        # % INPUTS:  
        # % phi = internal and solar gains;
        # % Hve = ventilation coefficients (ventilation and infiltration);
        # % theta_bound = external air temperature and sol-air temperature;
        # %onto the nodes (surface nodes AW and IW and air node);
        # % tau = time-step della simulazione.
        # % theta_set = set-point of considered thermal zone;
        # % theta_m0 = temperature of thermal mass of AW and IW building components 
        # %(grouped) at the previous hour; (initial if 1st time-step)
        
        # % OUTPUTS: 
        # % temperature nodes of the RC model: 
        # % theta_m_aw = x(0); thermal mass of AW building components
        # % theta_s_aw = x(1); surface of AW building components
        # % theta_lu_star = x(2); No physical meaning (node obtained from the
        # % delta-->star transformation)
        # % theta_I_lu = x(3); internal air temperature   <--- WHAT WE WILL EXTRACT AS OUTPUT outside the function     
        # % Q_hk_ges = x(4); heating/cooling load for maintaining the given setpoint temperature (*)  <--- WHAT WE WILL EXTRACT AS OUTPUT outside the function     
        # % theta_s_iw = x(5); surface of IW building components
        # % theta_m_iw = x(6); thermal mass of IW building components
        
        # % (*)Note: if  phi_HC > 0 HEATING LOAD; phi_HC < 0 COOLING LOAD.
        
        
        # % INPUT
        
        # % Resistances and capacitances of the 7R2C model
        R_lue_ve = 1e20 if H_ve[0] == 0 else 1/H_ve[0]
        R_lue_inf = 1e20 if H_ve[1] == 0 else  1/H_ve[1]
        
        Q_il_kon = phi[0]      # convective heat gains (interni) 
        Q_il_str_aw = phi[1]   # radiant heat gains on surface node AW (interni + solare) 
        Q_il_str_iw = phi[2]   # radiant heat gains on surface node IW (interni + solare) 
        
        theta_A_eq = theta_bound[0]   # equivalent outdoor temperature (sol-air temperature)
        theta_lue = theta_bound[1]     # outdoor air temperature 
       
        if flag == 'Tset':
            theta_I_lu = theta_set          # set-point interno di temperatura
                        
            # MATRIX OF THERMAL TRANSMITTANCES
            
            Y = np.zeros([6,6])
            
            Y[0,0] = -1/self.RrestAW - 1/self.R1AW - self.C1AW/tau
            Y[0,1] = 1/self.R1AW
            
            Y[1,0] = 1/self.R1AW
            Y[1,1] = -1/self.R1AW - 1/ self.RalphaStarAW
            Y[1,2] = 1/ self.RalphaStarAW
            Y[1,3] = self.sigma[1]
            
            Y[2,1] = 1/ self.RalphaStarAW
            Y[2,2] = -1/ self.RalphaStarAW - 1/self.RalphaStarIL -1/self.RalphaStarIW
            Y[2,4] = 1/self.RalphaStarIW
            
            Y[3,2] = 1/self.RalphaStarIL
            Y[3,3] = self.sigma[2]
            
            Y[4,2] = 1/self.RalphaStarIW
            Y[4,3] = self.sigma[0]
            Y[4,4] = -1/self.RalphaStarIW - 1/self.R1IW
            Y[4,5] = 1/self.R1IW
        
            Y[5,4] = 1/self.R1IW
            Y[5,5] = -1/self.R1IW - self.C1IW/tau
            
            
            # VECTOR OF KNOWN VALUES
            
            q = np.zeros([6,1])
            
            q[0] = -theta_A_eq/self.RrestAW - self.C1AW*self.theta_m0_vdi[0]/tau
            q[1] = -Q_il_str_aw
            q[2] = -theta_I_lu/self.RalphaStarIL
            q[3] = theta_I_lu/self.RalphaStarIL - Q_il_kon \
                    - (theta_lue - theta_I_lu)/R_lue_inf \
                    - (theta_sup - theta_I_lu)/R_lue_ve \
                    + self.Ca*(theta_I_lu - self.Ta0)/tau
            q[4] = -Q_il_str_iw
            q[5] = -self.C1IW*self.theta_m0_vdi[1]/tau
            
            
            # OUTPUT (UNKNOWN) VARIABLES OF THE LINEAR SYSTEM
            y = np.linalg.inv(Y).dot(q)
            return np.insert(y,3,theta_set)
            
        elif flag == 'phiset':
            # Note: the heat load in input is already distributed on the 3 nodes
            Q_hk_iw = Q_hk*self.sigma[0]        #% radiant heat flow from HVAC system (on surface node IW)
            Q_hk_aw = Q_hk*self.sigma[1]        #% radiant heat flow from HVAC system (on surface node AW)
            Q_hk_kon = Q_hk*self.sigma[2]       #% convective heat flow from HVAC system (on air node)
            
            # MATRIX OF THERMAL TRANSMITTANCES

            Y = np.zeros([6,6])  
            
            Y[0,0] = -1/self.RrestAW - 1/self.R1AW - self.C1AW/tau 
            Y[0,1] = 1/self.R1AW 
            
            Y[1,0] = 1/self.R1AW 
            Y[1,1] = -1/self.R1AW - 1/ self.RalphaStarAW 
            Y[1,2] = 1/ self.RalphaStarAW 
            					  
            
            Y[2,1] = 1/ self.RalphaStarAW 
            Y[2,2] = -1/ self.RalphaStarAW - 1/self.RalphaStarIL -1/self.RalphaStarIW 
            Y[2,3] = 1/self.RalphaStarIL 
            Y[2,4] = 1/self.RalphaStarIW 
            
            Y[3,2] = 1/self.RalphaStarIL 
            Y[3,3] = -1/self.RalphaStarIL  -1/R_lue_inf  -1/R_lue_ve  -self.Ca/tau 
            
            Y[4,2] = 1/self.RalphaStarIW 
            					 
            Y[4,4] = -1/self.RalphaStarIW - 1/self.R1IW 
            Y[4,5] = 1/self.R1IW 
            
            Y[5,4] = 1/self.R1IW 
            Y[5,5] = -1/self.R1IW - self.C1IW/tau 
            
            
            # VECTOR OF KNOWN TERMS
            
            q = np.zeros(6) 
            q[0] = -theta_A_eq/self.RrestAW - self.C1AW*self.theta_m0_vdi[0]/tau 
            q[1] = -Q_hk_aw - Q_il_str_aw 
            q[2] = 0
            q[3] = -Q_hk_kon - Q_il_kon - theta_lue/R_lue_inf - theta_sup/R_lue_ve -self.Ca*self.Ta0/tau
            q[4] = -Q_hk_iw - Q_il_str_iw 
            q[5] = -self.C1IW*self.theta_m0_vdi[1]/tau 
            
            # OUTPUT SISTEMA LINEARE
            
            y = np.linalg.inv(Y).dot(q)
            return np.insert(y,4,Q_hk)
        
        else:
            return 'wrong flag'
    
    
    def solveDD_Heating(self):
        
        T_e = -5
        if T_e < 0:
            p_extsat = 610.5*np.exp((21.875*T_e)/(265.5+T_e))
        else:
            p_extsat = 610.5*np.exp((17.269*T_e)/(237.3+T_e))
        RH_e = 0.8
        T_set_inv = 20
        RH_int_set_H = 0.6
        x_ext = 0.622*(RH_e*p_extsat/(self.p_atm-(RH_e*p_extsat)))
        G_v = 0
        G_da_inf = 0.2*self.rho_air*self.V/3600  
        Hve_inf = 0.1*self.cp_air*self.rho_air*self.V/3600
        G_da_vent = 0.5*self.rho_air*self.V/3600
        Hve_vent = 0.5*self.cp_air*self.rho_air*self.V/3600
        p_intsat = 610.5*np.exp((17.269*T_set_inv)/(237.3+T_set_inv))
        x_int_set = 0.622*(RH_int_set_H*p_intsat/(self.p_atm-(RH_int_set_H*p_intsat)))
        
        self.heatFlow_DDH = 1.2*(self.Htr_op + self.Htr_w)*(T_set_inv - T_e) + Hve_inf*(T_set_inv - T_e) + Hve_vent*(T_set_inv - T_e)

        self.latentFlow_DDH = (G_da_inf*(x_ext-x_int_set) + G_da_vent*(x_ext-x_int_set))*(self.r_0+self.cpv*T_set_inv) + G_v*(self.r_0+self.cpv*T_set_inv)
        self.latentFlow_DDH = -1*self.latentFlow_DDH
        
        
    def solveTZ(self,t,T_e,RH_e,p_extsat,P_DD,tau,mode = '1C'):
        
        flag_AHU = True
        P_max = P_DD[0]
        P_min = P_DD[1]
        
        '''Inizialization'''
        T_set_inv = self.schedules.heatingTSP[t]
        T_set_est =self.schedules.coolingTSP[t]
        RH_int_set_H = self.schedules.HeatingRHSP[t]
        RH_int_set_C = self.schedules.CoolingRHSP[t]
        # AHUOnOff = self.schedules.AHUOnOff[t]
        # AHUHUM = self.schedules.scalar_data['AHUHUM']
        self.T_sup[t] = self.schedules.AHUTSupp[t]
        x_sup = self.schedules.AHUxSupp[t]
        # Sens_Recovery_eff = self.schedules.scalar_data['sensRec']
        # Lat_Recovery_eff = self.schedules.scalar_data['latRec']
        # OutAirRatio = self.schedules.scalar_data['outdoorAirRatio']
        
        
        self.x_ext[t] = 0.622*(RH_e*p_extsat/(self.p_atm-(RH_e*p_extsat)))
        self.G_v[t] = self.schedules.vapour[t]
        
        '''Infiltration'''
        self.G_da_inf[t] = self.schedules.infFlowRate[t]*self.rho_air*self.V/3600  
        Hve_inf = self.schedules.infFlowRate[t]*self.cp_air*self.rho_air*self.V/3600
        
        '''Ventilation'''
        self.G_da_vent[t] =self.schedules.ventFlowRate[t]*self.rho_air*self.zone_area #/3600
        Hve_vent = self.schedules.ventFlowRate[t]*self.cp_air*self.rho_air*self.zone_area #/3600
        
        Hve = [Hve_vent , Hve_inf]
        
        # if mode == '1C':
        phi_load = [self.phi_ia[t], self.phi_st[t],self.phi_m[t]]
        # elif mode == '2C':
        #     phi_load = [self.Q_il_kon_I[t], self.Q_il_str_aw[t], self.Q_il_str_iw[t]]
        #     theta_bound = [self.theta_eq_tot[t],T_e]
        
        # if AHUOnOff  == 0:
        #     self.T_sup[t] = T_e
        #     x_sup = self.x_ext[t]
            
        
        '''SENSIBLE HEAT LOAD CALCULATION'''
        
        '''Heating mode'''
        
        # while flag_AHU:
        
         
        
        flag_plant_sens = self.schedules.plantOnOffSens[t]
        flag_plant_lat = self.schedules.plantOnOffLat[t]
        
        {
        1: self.heat_sens,
        -1: self.cool_sens, 
        0: self.no_sens          
        }[flag_plant_sens](t,T_e,RH_e,p_extsat,P_DD,tau,mode,P_max, P_min, T_set_inv, T_set_est, RH_int_set_H, RH_int_set_C, Hve, phi_load, x_sup)
        
        
        {
        1: self.heat_lat,
        -1: self.cool_lat, 
        0: self.no_lat          
        }[flag_plant_lat](t,T_e,RH_e,p_extsat,P_DD,tau,mode,P_max, P_min, T_set_inv, T_set_est, RH_int_set_H, RH_int_set_C, Hve, phi_load, x_sup)
        
        
        # if self.schedules.plantOnOffSens[t] == 1:
        
            
        #     if mode == '1C':
        #         pot, Ta, Ts, Tm = self.Sensible1C('Tset',Hve,T_set_inv, self.T_sup[t], T_e , phi_load,tau)
        #     elif mode == '2C':
        #         Tm_aw, Ts_aw, T_lu_star, Ta, pot, Ts_iw, Tm_iw = self.Sensible2C('Tset',phi_load, Hve, theta_bound, self.T_sup[t], tau, theta_set = T_set_inv)
            
        #     if pot>0 and pot<P_max*1000:
        #         self.heatFlow[t] = pot
        #         self.Air_temp[t] = Ta
        #         if mode == '1C':
        #             self.theta_m = Tm
        #         elif mode == '2C':
        #             self.theta_m_vdi = [Tm_aw,Tm_iw]
        #     else:
        #         if pot > P_max*1000:
        #             phi_set = P_max*1000
        #         else:
        #             phi_set = 0
                
        #         if mode == '1C':
        #             pot, Ta, Ts, Tm = self.Sensible1C('phiset',Hve,T_set_inv, self.T_sup[t], T_e , phi_load,tau, phi_HC_set = phi_set)
        #         elif mode == '2C':
        #             Tm_aw, Ts_aw, T_lu_star, Ta, pot, Ts_iw, Tm_iw = self.Sensible2C('phiset',phi_load, Hve, theta_bound, self.T_sup[t],  tau, Q_hk = phi_set)
        #         self.heatFlow[t] = pot
        #         self.Air_temp[t] = Ta
        #         if mode == '1C':
        #             self.theta_m = Tm
        #         elif mode == '2C':
        #             self.theta_m_vdi = [Tm_aw,Tm_iw]
                    
                   
        
        
        
        # '''Cooling mode'''
        # if self.schedules.plantOnOffSens[t] == -1:
            
        #     if mode == '1C':
        #         pot, Ta, Ts, Tm = self.Sensible1C('Tset',Hve,T_set_est, self.T_sup[t], T_e , phi_load,tau)
        #     elif mode == '2C':
        #         Tm_aw, Ts_aw, T_lu_star, Ta, pot, Ts_iw, Tm_iw = self.Sensible2C('Tset',phi_load, Hve, theta_bound,self.T_sup[t],  tau, theta_set = T_set_est)
            
        #     if pot<0 and pot>P_min*1000:
        #         self.heatFlow[t] = pot
        #         self.Air_temp[t] = Ta
        #         if mode == '1C':
        #             self.theta_m = Tm
        #         elif mode == '2C':
        #             self.theta_m_vdi = [Tm_aw,Tm_iw]    
        #     else:
        #         if pot < P_min*1000:
        #             phi_set = P_min*1000
        #         else:
        #             phi_set = 0
        #         if mode == '1C':
        #             pot, Ta, Ts, Tm = self.Sensible1C('phiset',Hve,T_set_est, self.T_sup[t], T_e , phi_load,tau, phi_HC_set = phi_set)
        #         elif mode == '2C':
        #             Tm_aw, Ts_aw, T_lu_star, Ta, pot, Ts_iw, Tm_iw = self.Sensible2C('phiset',phi_load, Hve, theta_bound,self.T_sup[t],  tau, Q_hk = phi_set)
        #         self.heatFlow[t] = pot
        #         self.Air_temp[t] = Ta
        #         if mode == '1C':
        #             self.theta_m = Tm
        #         elif mode == '2C':
        #             self.theta_m_vdi = [Tm_aw,Tm_iw]
        
        
        
        
        # '''Plant OFF'''
        # if self.schedules.plantOnOffSens[t] == 0:
            
        #     if mode == '1C':
        #         pot, Ta, Ts, Tm = self.Sensible1C('phiset',Hve,T_set_inv, self.T_sup[t], T_e , phi_load,tau)
        #     elif mode == '2C':
        #         Tm_aw, Ts_aw, T_lu_star, Ta, pot, Ts_iw, Tm_iw = self.Sensible2C('phiset',phi_load, Hve, theta_bound,self.T_sup[t],tau)
        #     self.heatFlow[t] = pot
        #     self.Air_temp[t] = Ta
        #     if mode == '1C':
        #         self.theta_m = Tm
        #     elif mode == '2C':
        #         self.theta_m_vdi = [Tm_aw,Tm_iw]          
        
        
        
        # '''LATENT HEAT LOAD CALCULATION'''
        
        # '''Dehumidification mode'''
        # if self.schedules.plantOnOffLat[t] == -1:
            
        #     if self.Air_temp[t] < 0:
        #         p_intsat = 610.5*np.exp((21.875*self.Air_temp[t])/(265.5+self.Air_temp[t]))
        #     else:
        #         p_intsat = 610.5*np.exp((17.269*self.Air_temp[t])/(237.3+self.Air_temp[t]))
            
        #     x_int_set = 0.622*(RH_int_set_C*p_intsat/(self.p_atm-(RH_int_set_C*p_intsat)))
        #     self.latentFlow[t] = (self.G_da_inf[t]*(self.x_ext[t]-x_int_set) + self.G_da_vent[t]*(x_sup-x_int_set)-self.rho_air*self.V*(x_int_set-self.x_m0)/tau)*(self.r_0+self.cpv*self.Air_temp[t]) + self.G_v[t]*(self.r_0+self.cpv*self.Air_temp[t])
            
        #     if self.latentFlow[t] < 0:
        #         self.latentFlow[t] = 0
        #         self.x_int[t] =(self.G_da_inf[t]*self.x_ext[t] + self.G_da_vent[t]*x_sup + self.G_v[t] + self.rho_air*self.V*self.x_m0/tau)/(self.G_da_inf[t] + self.G_da_vent[t] + self.rho_air*self.V/tau)
        #         p_int = self.p_atm*self.x_int[t]/(0.622+self.x_int[t])
        #         RH_int = p_int/p_intsat
        #         self.RH_i[t] = RH_int*100
               
        #     else:
        #         self.RH_i[t] = RH_int_set_C*100
        #         self.x_int[t] = x_int_set
        
        # '''Humidification mode'''
        # if self.schedules.plantOnOffLat[t] == 1:
            
        #     if self.Air_temp[t] < 0:
        #         p_intsat = 610.5*np.exp((21.875*self.Air_temp[t])/(265.5+self.Air_temp[t]))
        #     else:
        #         p_intsat = 610.5*np.exp((17.269*self.Air_temp[t])/(237.3+self.Air_temp[t]))
            
        #     x_int_set = 0.622*(RH_int_set_H*p_intsat/(self.p_atm-(RH_int_set_H*p_intsat)))
        #     self.latentFlow[t] = (self.G_da_inf[t]*(self.x_ext[t]-x_int_set) + self.G_da_vent[t]*(x_sup-x_int_set)-self.rho_air*self.V*(x_int_set-self.x_m0)/tau)*(self.r_0+self.cpv*self.Air_temp[t]) + self.G_v[t]*(self.r_0+self.cpv*self.Air_temp[t])
            
        #     if self.latentFlow[t] > 0:
        #         self.latentFlow[t] = 0
        #         self.x_int[t] =(self.G_da_inf[t]*self.x_ext[t] + self.G_da_vent[t]*x_sup + self.G_v[t] + self.rho_air*self.V*self.x_m0/tau)/(self.G_da_inf[t] + self.G_da_vent[t] + self.rho_air*self.V/tau)
        #         p_int = self.p_atm*self.x_int[t]/(0.622+self.x_int[t])
        #         RH_int = p_int/p_intsat
        #         self.RH_i[t] = RH_int*100
                
        #         if RH_int > 1:
        #             RH_int = 0.99
        #             self.RH_i[t] = RH_int*100
        #             p_int = p_intsat*RH_int
        #             self.x_int[t] = 0.622*(p_int/(self.p_atm-p_int))
                
        #     else:
        #         self.RH_i[t] = RH_int_set_H*100
        #         self.x_int[t] = x_int_set
        
        # '''Plant Off'''
        # if self.schedules.plantOnOffLat[t] == 0:
        #     if self.Air_temp[t] < 0:
        #         p_intsat = 610.5*np.exp((21.875*self.Air_temp[t])/(265.5+self.Air_temp[t]))
        #     else:
        #         p_intsat = 610.5*np.exp((17.269*self.Air_temp[t])/(237.3+self.Air_temp[t]))
        #         self.x_int[t] = (self.G_da_inf[t]*self.x_ext[t] + self.G_da_vent[t]*x_sup + self.G_v[t] + self.rho_air*self.V*self.x_m0/tau)/(self.G_da_inf[t] + self.G_da_vent[t] + self.rho_air*self.V/tau)
        #         p_int = self.p_atm*self.x_int[t]/(0.622+self.x_int[t])
        #         RH_int = p_int/p_intsat
        #         self.RH_i[t] = RH_int*100
                
        #         if RH_int > 1:
        #             RH_int = 0.99
        #             self.RH_i[t] = RH_int*100
        #             p_int = p_intsat*RH_int
        #             self.x_int[t] = 0.622*(p_int/(self.p_atm-p_int))
            
        #     self.latentFlow[t] = 0
        
        # self.latentFlow[t] = -1*self.latentFlow[t]
        
        '''AIR HANDLING UNIT HEAT DEMAND'''
        
        # self.ZoneAHU.AHUCalc(t,self.G_da_vent[t],AHUOnOff,AHUHUM,Sens_Recovery_eff,Lat_Recovery_eff,OutAirRatio,T_e,self.x_ext[t],self.Air_temp[t],self.x_int[t],self.T_sup[t],x_sup)
    
        '''Check if supply conditions are changed'''
        # err_T_sup = 0 #abs(self.T_sup[t] - self.ZoneAHU.T_supAHU[t])
        # err_x_sup = 0 #abs(x_sup - self.ZoneAHU.x_sup)
        # if err_T_sup > 0.1 or err_x_sup > 0.0001:
        #     self.T_sup[t] = self.ZoneAHU.T_supAHU[t]
        #     x_sup = self.ZoneAHU.x_sup
        # else:
        # if mode == '1C':
        self.theta_m0 = self.theta_m
        self.T_wall_0 = T_e + (self.theta_m - T_e)*self.Htr_em/self.H_lim_ext_wall_1C
        # elif mode == '2C':
        #     self.theta_m0_2c[t] = self.theta_m_vdi
        #     self.ext_surf_heat_gain[t] = self.C1AW/tau*(self.theta_m_vdi[0]-self.theta_m0_vdi[0])
        #     self.theta_m0_vdi = self.theta_m_vdi
        #     self.T_wall_0 = self.theta_eq_tot[t] + (Tm_aw - self.theta_eq_tot[t])*self.R_lim_ext_wall_2C/self.RrestAW
        self.Ta0 = self.Air_temp[t]
        self.x_m0 = self.x_int[t]
        self.T_wall_0_vector[t] = self.T_wall_0
        # flag_AHU = False

                
    def reset_init_values(self):
        
        ''' This method allows to reset temperaturest starting values after
            the DesignDays calculation and before running simulation'''   
            
        self.Ta0 = 15
        self.theta_m0 = 15
        self.theta_m0_vdi = [15, 15]                                           # aw the first, iw the second
        
    
    def heat_sens(self, t,T_e,RH_e,p_extsat,P_DD,tau,mode,P_max, P_min, T_set_inv, T_set_est, RH_int_set_H, RH_int_set_C, Hve, phi_load, x_sup):
        # if mode == '1C':
        pot, Ta, Ts, Tm = self.Sensible1C('Tset',Hve,T_set_inv, self.T_sup[t], T_e , phi_load,tau)
        # elif mode == '2C':
        #     Tm_aw, Ts_aw, T_lu_star, Ta, pot, Ts_iw, Tm_iw = self.Sensible2C('Tset',phi_load, Hve, theta_bound, self.T_sup[t], tau, theta_set = T_set_inv)
        
        if pot>0 and pot<P_max*1000:
            self.heatFlow[t] = pot
            self.Air_temp[t] = Ta
            # if mode == '1C':
            self.theta_m = Tm
            # elif mode == '2C':
            #     self.theta_m_vdi = [Tm_aw,Tm_iw]
        else:
            if pot > P_max*1000:
                phi_set = P_max*1000
            else:
                phi_set = 0
            
            # if mode == '1C':
            pot, Ta, Ts, Tm = self.Sensible1C('phiset',Hve,T_set_inv, self.T_sup[t], T_e , phi_load,tau, phi_HC_set = phi_set)
            # elif mode == '2C':
            #     Tm_aw, Ts_aw, T_lu_star, Ta, pot, Ts_iw, Tm_iw = self.Sensible2C('phiset',phi_load, Hve, theta_bound, self.T_sup[t],  tau, Q_hk = phi_set)
            self.heatFlow[t] = pot
            self.Air_temp[t] = Ta
            # if mode == '1C':
            self.theta_m = Tm
            # elif mode == '2C':
            #     self.theta_m_vdi = [Tm_aw,Tm_iw]
        
    def cool_sens(self,t,T_e,RH_e,p_extsat,P_DD,tau,mode,P_max, P_min, T_set_inv, T_set_est, RH_int_set_H, RH_int_set_C, Hve, phi_load, x_sup):
        # if mode == '1C':
        pot, Ta, Ts, Tm = self.Sensible1C('Tset',Hve,T_set_est, self.T_sup[t], T_e , phi_load,tau)
        # elif mode == '2C':
        #     Tm_aw, Ts_aw, T_lu_star, Ta, pot, Ts_iw, Tm_iw = self.Sensible2C('Tset',phi_load, Hve, theta_bound,self.T_sup[t],  tau, theta_set = T_set_est)
        
        if pot<0 and pot>P_min*1000:
            self.heatFlow[t] = pot
            self.Air_temp[t] = Ta
            # if mode == '1C':
            self.theta_m = Tm
            # elif mode == '2C':
            #     self.theta_m_vdi = [Tm_aw,Tm_iw]    
        else:
            if pot < P_min*1000:
                phi_set = P_min*1000
            else:
                phi_set = 0
            # if mode == '1C':
            pot, Ta, Ts, Tm = self.Sensible1C('phiset',Hve,T_set_est, self.T_sup[t], T_e , phi_load,tau, phi_HC_set = phi_set)
            # elif mode == '2C':
            #     Tm_aw, Ts_aw, T_lu_star, Ta, pot, Ts_iw, Tm_iw = self.Sensible2C('phiset',phi_load, Hve, theta_bound,self.T_sup[t],  tau, Q_hk = phi_set)
            self.heatFlow[t] = pot
            self.Air_temp[t] = Ta
            # if mode == '1C':
            self.theta_m = Tm
            # elif mode == '2C':
            #     self.theta_m_vdi = [Tm_aw,Tm_iw]
    
    def no_sens(self, t,T_e,RH_e,p_extsat,P_DD,tau,mode,P_max, P_min, T_set_inv, T_set_est, RH_int_set_H, RH_int_set_C, Hve, phi_load, x_sup):
        # if mode == '1C':
        pot, Ta, Ts, Tm = self.Sensible1C('phiset',Hve,T_set_inv, self.T_sup[t], T_e , phi_load,tau)
        # elif mode == '2C':
        #     Tm_aw, Ts_aw, T_lu_star, Ta, pot, Ts_iw, Tm_iw = self.Sensible2C('phiset',phi_load, Hve, theta_bound,self.T_sup[t],tau)
        self.heatFlow[t] = pot
        self.Air_temp[t] = Ta
        # if mode == '1C':
        self.theta_m = Tm
        # elif mode == '2C':
        #     self.theta_m_vdi = [Tm_aw,Tm_iw]       
    
    def heat_lat(self, t,T_e,RH_e,p_extsat,P_DD,tau,mode,P_max, P_min, T_set_inv, T_set_est, RH_int_set_H, RH_int_set_C, Hve, phi_load, x_sup):
            # if self.Air_temp[t] < 0:
            #     p_intsat = 610.5*np.exp((21.875*self.Air_temp[t])/(265.5+self.Air_temp[t]))
            # else:
            p_intsat = 610.5*np.exp((17.269*self.Air_temp[t])/(237.3+self.Air_temp[t]))
            
            x_int_set = 0.622*(RH_int_set_H*p_intsat/(self.p_atm-(RH_int_set_H*p_intsat)))
            self.latentFlow[t] = (self.G_da_inf[t]*(self.x_ext[t]-x_int_set) + self.G_da_vent[t]*(x_sup-x_int_set)-self.rho_air*self.V*(x_int_set-self.x_m0)/tau)*(self.r_0+self.cpv*self.Air_temp[t]) + self.G_v[t]*(self.r_0+self.cpv*self.Air_temp[t])
            
            if self.latentFlow[t] > 0:
                self.latentFlow[t] = 0
                self.x_int[t] =(self.G_da_inf[t]*self.x_ext[t] + self.G_da_vent[t]*x_sup + self.G_v[t] + self.rho_air*self.V*self.x_m0/tau)/(self.G_da_inf[t] + self.G_da_vent[t] + self.rho_air*self.V/tau)
                p_int = self.p_atm*self.x_int[t]/(0.622+self.x_int[t])
                RH_int = p_int/p_intsat
                self.RH_i[t] = RH_int*100
                
                if RH_int > 1:
                    RH_int = 0.99
                    self.RH_i[t] = RH_int*100
                    p_int = p_intsat*RH_int
                    self.x_int[t] = 0.622*(p_int/(self.p_atm-p_int))
                
            else:
                self.RH_i[t] = RH_int_set_H*100
                self.x_int[t] = x_int_set
        
    
    def cool_lat(self, t,T_e,RH_e,p_extsat,P_DD,tau,mode,P_max, P_min, T_set_inv, T_set_est, RH_int_set_H, RH_int_set_C, Hve, phi_load, x_sup):
        # if self.Air_temp[t] < 0:
        #     p_intsat = 610.5*np.exp((21.875*self.Air_temp[t])/(265.5+self.Air_temp[t]))
        # else:
        p_intsat = 610.5*np.exp((17.269*self.Air_temp[t])/(237.3+self.Air_temp[t]))
        
        x_int_set = 0.622*(RH_int_set_C*p_intsat/(self.p_atm-(RH_int_set_C*p_intsat)))
        self.latentFlow[t] = (self.G_da_inf[t]*(self.x_ext[t]-x_int_set) + self.G_da_vent[t]*(x_sup-x_int_set)-self.rho_air*self.V*(x_int_set-self.x_m0)/tau)*(self.r_0+self.cpv*self.Air_temp[t]) + self.G_v[t]*(self.r_0+self.cpv*self.Air_temp[t])
        
        if self.latentFlow[t] < 0:
            self.latentFlow[t] = 0
            self.x_int[t] =(self.G_da_inf[t]*self.x_ext[t] + self.G_da_vent[t]*x_sup + self.G_v[t] + self.rho_air*self.V*self.x_m0/tau)/(self.G_da_inf[t] + self.G_da_vent[t] + self.rho_air*self.V/tau)
            p_int = self.p_atm*self.x_int[t]/(0.622+self.x_int[t])
            RH_int = p_int/p_intsat
            self.RH_i[t] = RH_int*100
           
        else:
            self.RH_i[t] = RH_int_set_C*100
            self.x_int[t] = x_int_set
       
        
    def no_lat(self, t,T_e,RH_e,p_extsat,P_DD,tau,mode,P_max, P_min, T_set_inv, T_set_est, RH_int_set_H, RH_int_set_C, Hve, phi_load, x_sup):
        # if self.Air_temp[t] < 0:
        #     p_intsat = 610.5*np.exp((21.875*self.Air_temp[t])/(265.5+self.Air_temp[t]))
        # else:
        p_intsat = 610.5*np.exp((17.269*self.Air_temp[t])/(237.3+self.Air_temp[t]))
        self.x_int[t] = (self.G_da_inf[t]*self.x_ext[t] + self.G_da_vent[t]*x_sup + self.G_v[t] + self.rho_air*self.V*self.x_m0/tau)/(self.G_da_inf[t] + self.G_da_vent[t] + self.rho_air*self.V/tau)
        p_int = self.p_atm*self.x_int[t]/(0.622+self.x_int[t])
        RH_int = p_int/p_intsat
        self.RH_i[t] = RH_int*100
        
        if RH_int > 1:
            RH_int = 0.99
            self.RH_i[t] = RH_int*100
            p_int = p_intsat*RH_int
            self.x_int[t] = 0.622*(p_int/(self.p_atm-p_int))
        
        self.latentFlow[t] = 0
    
#%%        

class Building:
    
    '''
    building class
    generates the single thermal zone or multifloor thermal zone
    
    input:
        list of zone surfaces (only list of list of vertices, not surface class. 
                               The class will be initiatilized in this class)
        envelope id
        schedule id
    '''
    
    Rat_iw = 2.5
    eps_se = 0.9
    h_r = 5*eps_se
    
    def __init__(self, archetype, end_use,
                 envelopes, heating_plant, cooling_plant,weather):
        
        self.archetype = archetype
        '''
        one thermal zone for building
        ---
        THIS IS FOR LOD1
        ---
        '''
        ts = weather.ts
        hours = weather.hours
        azSubdiv = weather.azSubdiv
        hSubdiv = weather.hSubdiv
        rh_gross = 1.
        rh_net = 1.
        self.weather = weather
        '''Output vectors inizialization'''
        self.l = ts*(hours-1) + 1
        self.Air_tempBD = np.zeros(self.l)
        self.RH_iBD = np.zeros(self.l)
        self.heatFlowBD = np.zeros(self.l)
        self.latentFlowBD = np.zeros(self.l)
        self.AHUDemandBD = np.zeros(self.l)
        self.AHUDemand_latBD = np.zeros(self.l)
        self.AHUDemand_sensBD = np.zeros(self.l)
        
        self.__flagCheckWalls = False
        self.name = self.archetype.name
        self.end_use = end_use
        self.footprint = self.archetype.floorArea 
        self.oneFloorHeight = 3.3 
        self.orientation = self.archetype.orientation
        self.area_fin = self.archetype.area_fin # area totale finestre archetipo
        if self.area_fin/self.footprint < 1/8:
            self.area_fin = self.footprint*1/8        
        nSurf_exp = self.orientation[0] + self.orientation[1] + self.orientation[2] + self.orientation[3]
        glazed_area_per_orientation = self.area_fin/nSurf_exp
        self.zones={}
        self.buildingSurfaces = {}

        i = 0
        
        # self.hmax = []
        # self.hmin = []
        self.extWallArea = 0
        self.extRoofArea = 0
        self.extWallWinArea = 0
        self.extWallOpaqueArea = 0
        self.T_out_inf = 15.
        self.T_wall_0 = 15.
        self.G_inf_0 = 0.
        self.T_out_AHU = 15.
        self.G_vent_0 = 0.
        self.H_waste = 0.
        
        
        
        self.nFloors = self.archetype.nFloors            
            
        surfList = self.archetype.surfList
        
        for surf in surfList:
            '''
            self.wwr = envelopes[archId].Window.wwr
            i += 1
            surface = Surface('Building Surface '+str(i),azSubdiv,hSubdiv,self.wwr,rh_gross,surf)
            '''
            
            
            # TRASCURO WWR PER IL CALCOLO DEL GLAZED-AREA
            # self.wwr = envelopes[self.archetype.archID].Window.wwr
            i += 1
            # surface = Surface('Building Surface '+str(i),self.nFloors*self.footprint,
            #                   self.orientation,azSubdiv,hSubdiv,self.wwr,rh_gross,surf)

            surface = Surface('Building Surface '+str(i),self.nFloors*self.footprint,
                              self.orientation, glazed_area_per_orientation,azSubdiv,hSubdiv,rh_gross,surf)

            
            # if surface.type == 'GroundFloor':
            #     self.footprint = footprint
            #     #self.footprint += surface.area
            if surface.type == 'ExtWall':
                self.extWallWinArea += round(surface.glazedArea,2)
                self.extWallOpaqueArea += round(surface.opaqueArea,2)
                # self.hmax.append(surface.maxHeight())
                # self.hmin.append(surface.minHeight())
            if surface.type == 'Roof':
                self.extRoofArea += round(surface.area,2)
            self.buildingSurfaces['Building Surface '+str(i)]=surface
            self.ii = i
  

        # self.hmax = np.mean(self.hmax)
        # self.hmin = np.mean(self.hmin)
        # self.checkExtWallsMod2()
        # self.buildingHeight = self.hmax - self.hmin
        self.Vertsurf = []
        for surf in self.buildingSurfaces.values():
            if surf.type == 'ExtWall':
                self.extWallArea += round(surf.area,2)
                # surf.surfHeight = self.buildingHeight
                # self.Vertsurf.append([surf,[]])
        

        if self.footprint == 0:
            self.footprint = 1.
        self.total_area = round((self.nFloors)*self.footprint,2)
        self.Volume = round(rh_net*(self.oneFloorHeight*self.footprint*self.nFloors),2)
        if self.Volume == 0:
            self.Volume = 0.0001
        self.archId = str(self.archetype.archID)
        self.H_plant_type = heating_plant
        self.C_plant_type = cooling_plant
        self.BDPlant = Plants(self.H_plant_type,self.C_plant_type,self.l,ts)
        self.Pnom_H_BD = 1e20
        self.Pnom_C_BD = -1e20

        
    def BDParamsandLoads(self,model,envelopes,sched,AdSurf = [],splitInZone=False):
        
        '''
        This method allows to initialize thermal zones and to calculate
        Parameters and Internal and Solar loads for each zone of the buildings
        '''
        if not splitInZone:
            i = 1
            for ads in AdSurf:
                self.buildingSurfaces['Building Surface '+str(self.ii+i)] = SurfaceInternalMass(('Building Surface '+str(self.ii+i)),          #name
                                   ads[1],                                                                                                    #area
                                   surfType=ads[0])                                                                                           #surftype=''
                i += 1
            
            '''
            self.buildingSurfaces['Building Surface '+str(self.ii+1)] = SurfaceInternalMass(('Building Surface '+str(self.ii+1)),(self.total_area-self.footprint),surfType='IntCeiling')
            self.buildingSurfaces['Building Surface '+str(self.ii+2)] = SurfaceInternalMass(('Building Surface '+str(self.ii+1)),(self.total_area-self.footprint),surfType='IntFloor')
            self.buildingSurfaces['Building Surface '+str(self.ii+3)] = SurfaceInternalMass(('Building Surface '+str(self.ii+2)),self.Rat_iw*self.nFloors*self.footprint,surfType='IntWall')
            '''
            
            self.zones['Zone'] = ThermalZone(1,self.name,
                                 envelopes[self.archId],
                                 sched,
                                 self.buildingSurfaces.values(),
                                 self.Volume,
                                 self.footprint*self.nFloors,
                                 self.l)
            
            if (model == '1C') or (model == 'QSS'):
                self.zones['Zone'].zoneParameter13790()
                self.zones['Zone'].calculate_zone_loads_ISO13790(self.weather.SolarGains,
                                                                 self.weather.dT_er,
                                                                 self.eps_se,
                                                                 self.h_r)
            
            elif model == '2C':
                self.zones['Zone'].zoneParameter6007()
                self.zones['Zone'].calculate_zone_loads_vdi6007(self.weather.Text,
                                                                self.weather.SolarGains)               
            
            else:
                sys.exit('Give a proper model: "1C", "2C" or "QSS" ')
        
        self.phi_int_medio = self.zones['Zone'].phi_int_medio
        self.phi_sol_medio = self.zones['Zone'].phi_sol_medio
        self.phi_int_medio_spec = self.zones['Zone'].phi_int_medio / self.footprint
        self.phi_sol_medio_spec = self.zones['Zone'].phi_sol_medio / self.footprint
        
        
        # self.BuildPV = DistrictPVGIS(self.extRoofArea,self.l)
        # self.BuildPV.PV_calc(self.weather.SolarGains,
        #                      self.weather.Text,self.weather.w)
        
    def BDdesigndays_Heating(self,Plant_calc):
        '''
        This method allows to calculate Heating maximum power required by
        buildings during design days
        '''
        for z in self.zones.values():
            z.solveDD_Heating()
            self.Pnom_H_BD = (self.zones['Zone'].heatFlow_DDH + self.zones['Zone'].latentFlow_DDH)/1000  # [kW]
            self.PDesignH = self.Pnom_H_BD
            
        if Plant_calc == 'NO':
            self.Pnom_H_BD = 1e20
        
    
    def BDdesigndays_Cooling(self,t,T_e,RH_e,p_extsat,tau,Plant_calc,model = '1C'):
        '''
        This method allows to calculate Cooling maximum power required by
        buildings during design days
        '''
        # for z in self.zones.values():
        self.zones['Zone'].solveTZ(t,T_e,RH_e,p_extsat,1e20*np.array([1,-1]),tau,model)
        self.Pnom_C_BD = min((self.zones['Zone'].heatFlow - self.zones['Zone'].latentFlow)/1000 + self.zones['Zone'].ZoneAHU.AHUDemand)   # [kW]
        self.PDesignC = self.Pnom_C_BD
            
        # if Plant_calc == 'NO':
        #     self.Pnom_C_BD = -1e20


    def BDplants(self,Plants_list,T_ext_H_avg):
        
        '''
        This method allows to set the plant of each building and to
        check minimum plant efficiency
        '''
        self.BDPlant.setPlant(self.H_plant_type,self.C_plant_type,Plants_list,self.Pnom_H_BD,self.Pnom_C_BD,T_ext_H_avg)
        
        
    def solve(self,t,Plants_list,T_e,RH_e,p_extsat,tau,Plant_calc,model = '1C'):
        
        #☻ for z in self.zones.values():
        self.zones['Zone'].solveTZ(t,T_e,RH_e,p_extsat,[self.Pnom_H_BD,self.Pnom_C_BD],tau,model)
        self.Air_tempBD[t] = self.zones['Zone'].Air_temp[t]
        self.RH_iBD[t] = self.zones['Zone'].RH_i[t]
        self.heatFlowBD[t] = self.zones['Zone'].heatFlow[t]/1000           # [kW]
        self.latentFlowBD[t] = -1 * self.zones['Zone'].latentFlow[t]/1000       # [kW]
        self.AHUDemandBD[t] = self.zones['Zone'].ZoneAHU.AHUDemand[t]      # [kW]
        self.AHUDemand_latBD[t] = self.zones['Zone'].ZoneAHU.AHUDemand_lat[t]
        self.AHUDemand_sensBD[t] = self.zones['Zone'].ZoneAHU.AHUDemand_sens[t]
        
        self.T_out_inf = np.mean(self.Air_tempBD[t])                           # for Urban Canyon
        self.T_wall_0 = np.mean(self.zones['Zone'].T_wall_0)                   # for Urban Canyon
        self.G_inf_0 = self.zones['Zone'].G_da_inf[t] / self.zones['Zone'].rho_air
        self.T_out_AHU = self.zones['Zone'].ZoneAHU.T_out
        self.G_vent_0 = self.zones['Zone'].G_da_vent[t] / self.zones['Zone'].rho_air
        
        # if Plant_calc == 'YES':
        #     self.BDPlant.solvePlant(self.H_plant_type,self.C_plant_type,Plants_list,self.heatFlowBD[t] + self.latentFlowBD[t] + self.AHUDemandBD[t],t,T_e,self.Air_tempBD[t],self.RH_iBD[t])
        #     self.H_waste = self.BDPlant.H_waste[t]
        # elif Plant_calc == 'NO':
            # pass
            
    def checkExtWalls(self):
        
        '''
        This method checks if there are coincident external walls
        and, in case, corrects their area
        '''
        #n=0
        if not self.__flagCheckWalls:
            self.__flagCheckWalls = True
            for surf in self.buildingSurfaces.values():
                if surf.type=='GroundFloor':
                    for i in range(len(surf.vertList)):
                        v1=surf.vertList[i-1]
                        v2=surf.vertList[i]
                        coincidentWalls=[]
                        for wall in self.buildingSurfaces.values():
                            if wall.type=='ExtWall':
                                if (v1 in wall.vertList and v2 in wall.vertList):
                                    coincidentWalls.append(wall)
                        if len(coincidentWalls)==2:
                            #n += 1
                            #surf.printInfo()
                            #coincidentWalls[0].printInfo()
                            #coincidentWalls[1].printInfo()
                            if coincidentWalls[0].area >= coincidentWalls[1].area:
                                coincidentWalls[0].area = coincidentWalls[0].area - coincidentWalls[1].area
                                coincidentWalls[1].area = 0
                            else:
                                coincidentWalls[1].area = coincidentWalls[1].area - coincidentWalls[0].area
                                coincidentWalls[0].area = 0
                                
                        
                        if len(coincidentWalls)>2:
                            print('WARNING: building "'+self.name+'" could have 3 or more coincident walls')
                            [i.printInfo() for i in coincidentWalls]
    
    
    def checkExtWallsMod(self):
        '''
        This method checks if there are coincident external walls
        and, in case, corrects their area
        '''
        #n=0
        if not self.__flagCheckWalls:
            self.__flagCheckWalls = True
            for wall1 in self.buildingSurfaces.values():
                if wall1.type=='ExtWall':
                    for wall2 in self.buildingSurfaces.values():
                        if (wall2.type=='ExtWall' and wall2 != wall1 and wall1.checkSurfaceCoincidence(wall2)):
                            
                            if wall1.area >= wall2.area:
                                wall1.area = wall1.area - wall2.area
                                wall2.area = 0
                            else :
                                wall2.area = wall2.area - wall1.area
                                wall1.area = 0 
    
    
    def checkExtWallsMod2(self):
        '''
        This method checks if there are coincident external walls
        and, in case, corrects their area
        '''
        #n=0
        if not self.__flagCheckWalls:
            self.__flagCheckWalls = True
            
            for wall1 in self.buildingSurfaces.values():
                if wall1.type=='ExtWall':
                    for wall2 in self.buildingSurfaces.values():
                        if (wall2.type=='ExtWall' and wall2 != wall1 and wall1.checkSurfaceCoincidence(wall2)):
                            
                            #print(wall1.name,wall2.name)
                            intersectionArea = wall1.calculateIntersectionArea(wall2)
                            wall1.reduceArea(intersectionArea)
                            '''
                            wall2.correctArea(wall2.Area - intersectionArea)
                            this line is not necessary, because the first for cycle will consider wall1 firstly, then after a while wall2
                            and will reduce wall1 and wall2 area in two distinct moments
                            '''
            
            
    def plotBuilding(self,addSurf):
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.view_init(elev=90,azim=0)
        for surface in self.buildingSurfaces.values():
            if surface.type=='ExtWall':
                x_v=np.zeros(0)
                y_v=np.zeros(0)
                z_v=np.zeros(0)
                for vert in surface.vertList:
                    x_v=np.append(x_v,vert[0])
                    y_v=np.append(y_v,vert[1])
                    z_v=np.append(z_v,vert[2])
    
                ax.plot(x_v,y_v,z_v, 'b-.x')
                
        x_v=np.zeros(0)
        y_v=np.zeros(0)
        z_v=np.zeros(0)       
        for vert in addSurf.vertList:
            x_v=np.append(x_v,vert[0])
            y_v=np.append(y_v,vert[1])
            z_v=np.append(z_v,vert[2])
        ax.plot(x_v,y_v,z_v, 'r-.x')
    
    
    def printBuildingInfo(self):
        print('nome '+str(self.name)+
              '\nfootprint: '+str(self.footprint)+
              '\nN_floors: '+str(self.nFloors)+
              '\nheight: '+str(self.buildingHeight)+
              '\nexternal wall area: '+str(self.extWallArea)
              +'\n')
        
    def output_df(self):
        import pandas as pd
        df = pd.DataFrame(columns = ['AHU (kW)',
                                     'ElEn (kW)',
                                     'Gas (Nm3)',
                                     'HeatFLow (kW)',
                                     'LatentFlow (kW)',
                                     'RH (%)',
                                     'Temp (°C)',
                                     'App (W)',
                                     'Light (W)',
                                     'Occ (W)'])
        
        df['AHU (kW)'] = self.AHUDemandBD
        df['ElEn (kW)'] = self.BDPlant.Electrical_energy_consumption
        df['Gas (Nm3)'] = self.BDPlant.Gas_consumption
        df['HeatFLow (kW)'] = self.heatFlowBD
        df['LatentFlow (kW)'] = self.latentFlowBD
        df['RH (%)'] = self.RH_iBD
        df['Temp (°C)'] = self.Air_tempBD
        df['App (W)'] = self.zones['Zone'].schedules.people
        df['Light (W)'] = self.zones['Zone'].schedules.appliances
        df['Occ (W)'] = self.zones['Zone'].schedules.lighting
        df['Vap (kg/s)'] = self.zones['Zone'].schedules.vapour
        
        return df
    
    def output_df_light(self):
        import pandas as pd
        df = pd.DataFrame(columns = ['HeatFLow (kW)',
                                     'LatentFlow (kW)',
                                     'Temp (°C)'])
        
        #df['ElEn (kW)'] = self.BDPlant.Electrical_energy_consumption
        df['HeatFLow (kW)'] = self.heatFlowBD
        df['LatentFlow (kW)'] = self.latentFlowBD
        #df['RH (%)'] = self.RH_iBD
        df['Temp (°C)'] = self.Air_tempBD
        
        return df
    
    def solve_QuasiSteadyState(self, Tset_H = 20, Text_m = None):
        
        self.Htr = self.zones['Zone'].Htr_op + self.zones['Zone'].Htr_w  # W/K
        infFlowRate = self.zones['Zone'].schedules.infFlowRate[1]
        self.Hve = infFlowRate*self.zones['Zone'].cp_air*self.zones['Zone'].rho_air*self.zones['Zone'].V/3600  # W/K   #+ self.zones['Zone'].Hve_vent
        # Te_m      = self.Text_m
        DD_m      = Tset_H - Text_m
        phi_sol_m = self.monthly_gains(self.zones['Zone'].phi_sol)
        phi_int_m = self.monthly_gains(self.zones['Zone'].phi_int)
        self.Qh_ht = (self.Htr + self.Hve)*24/1000*DD_m 
        self.Qh_gn = (phi_sol_m+phi_int_m)
        self.eta_H, self.eta_C = self.rendimenti_carichi_gratuiti()
        self.Qh_nd = self.Qh_ht - self.eta_H*self.Qh_gn
        
        return
    
    def monthly_gains(self, phi):
        phi_m = np.zeros(12)
        phi_m[0] = np.sum(phi[:31*24]) # january
        phi_m[1] = np.sum(phi[31*24+1:(31+28)*24]) # feb
        phi_m[2] = np.sum(phi[(31+28)*24+1:(31+28+31)*24]) # march
        phi_m[3] = np.sum(phi[(31+28+31)*24+1:(31+28+31+30)*24]) # april
        phi_m[4] = np.sum(phi[(31+28+31+30)*24+1:(31+28+31+30+31)*24]) # may
        phi_m[5] = np.sum(phi[(31+28+31+30+31)*24+1:(31+28+31+30+31+30)*24]) # june
        phi_m[6] = np.sum(phi[(31+28+31+30+31+30)*24+1:(31+28+31+30+31+30+31)*24]) # july
        phi_m[7] = np.sum(phi[(31+28+31+30+31+30+31)*24+1:(31+28+31+30+31+30+31+31)*24]) #august
        phi_m[8] = np.sum(phi[(31+28+31+30+31+30+31+31)*24+1:(31+28+31+30+31+30+31+31+30)*24]) # sept
        phi_m[9] = np.sum(phi[(31+28+31+30+31+30+31+31+30)*24+1:(31+28+31+30+31+30+31+31+30+31)*24])  #oct
        phi_m[10] = np.sum(phi[(31+28+31+30+31+30+31+31+30+31)*24+1:(31+28+31+30+31+30+31+31+30+31+30)*24])  #nov
        phi_m[11] = np.sum(phi[(31+28+31+30+31+30+31+31+30+31+30)*24+1:365*24])  #dic
        return phi_m
    
    def rendimenti_carichi_gratuiti(self):
        # Qui bisogna fare il calcolo secondo norma 
        # gamma_H = np.divide(self.Qh_ht,self.Qh_gn)
        # a_H0 = 1.
        # tau_H0 = 15.
        # tau = (self.C_m/3600)/(self.Htr + self.Hve)
        # a_H = a_H0 + tau/tau_H0 
        # if ((gamma_H > 0) and (gamma_H < 1)):
        #     eta_H = (1-gamma_H**a_H)/(1-gamma_H**(a_H+1))
        # elif gamma_H == 1:
        #     eta_H = a_H/(a_H + 1)
        # elif gamma_H < 0:
        #     eta_H = 1/gamma_H
                                     
        eta_H = 0.7
        eta_C = 0.7
        return eta_H, eta_C
        
        
    
#%%

class Complex:
    
    ''' Complex class in order to merge the output vectors of the buildings
    belonging to the same complex'''
    
    def __init__(self, Complexname):
        
        '''Vectors inizialization'''
        year = 8760
        self.ComplexName = Complexname
        self.heatFlowC = np.zeros(year)
        self.latentFlowC = np.zeros(year)
        self.AHUDemandC = np.zeros(year)
        self.AHUDemand_latC = np.zeros(year)
        self.AHUDemand_sensC = np.zeros(year)
        

 #%%        
# '''
# TEST METHOD
# '''
# import os

# if __name__ == '__main__':
#     buildingName = 'edificio Prova'
#     archId = 1
    
#     env_path = os.path.join('..','Input','buildings_envelope_V02_EP.xlsx')
#     envelopes = loadEvelopes(env_path)
    
#     # sched_path = os.path.join('..','Input','Schedule_ICA-RC.xlsx')
#     # Archetype = pd.read_excel(sched_path,sheet_name="Archetype",header=[1],index_col=[0])
#     # Arch = Archetype.loc['Office']
#     # schedule = pd.read_excel(sched_path,sheet_name="Schedule",usecols="C:end",header=[2])
    
#     iopath = os.path.join('..','Input', 'ITA_Venezia-Tessera.161050_IGDG.epw')
#     year=2007
#     epw_0=pvlib.iotools.read_epw(iopath,coerce_year=year)
#     epw = epw_0[0].reset_index(drop=True)
#     #time = epw[0].index
#     time = np.arange(8760)
#     tz='Europe/Rome'
#     T_ext = epw['temp_air']
#     ghi_infrared = epw['ghi_infrared']
#     Sigma = 5.6697e-08
#     T_as = (ghi_infrared/Sigma)**0.25-(273+T_ext)
#     dT_er = statistics.mean(T_ext-T_as)
#     lat, lon = epw_0[1]['latitude'], epw_0[1]['longitude']
#     site = pvlib.location.Location(lat, lon, tz=tz)
#     SolarGains = pd.read_csv(os.path.join('..','Input','PlanesIrradiances.csv'),header=[0,1,2],index_col=[0]).set_index(time)
#     rho_air = 1.2   #kg/m3
#     cp_air = 1000   #[J/(kg K)]
    
#     schedpath2 = os.path.join('..','Input','Schedule_ICA-RC.xlsx')
#     sched2 = loadArchetype(schedpath2,time)
#     sched_zone = sched2['Office']
    
#     #zone1 = ThermalZone(1, building)
#     #zone2 = ThermalZone(2, building)
    
#     #wall1=Surface([[0,0,0],[1,0,0],[1,0,9.5],[0,0,9.5]])
#     #wall2=Surface([[0,0,0],[0,0,9.5],[0,1,9.5],[0,1,0]])
#     #wall3=Surface([[0,1,0],[0,1,9.5],[1,1,9.5],[1,1,0]])
#     #wall4=Surface([[1,0,0],[1,1,0],[1,1,9.5],[1,0,9.5]])
    
#     #floorPT = Surface([[0,0,0],[0,1,0],[1,1,0],[1,0,0]])
#     #roofP1 = Surface([[0,0,9.5],[1,0,9.5],[1,1,9.5],[0,1,9.5]])
    
#     #intWall = SurfaceInternalMass(2)
#     #celinig1 = SurfaceInternalAdjacent(2,adjacentZone = zone2)
#     #celinig2 = SurfaceInternalAdjacent(2,adjacentZone = zone1)
    
#     #surfaces1=[wall1,wall2,wall3,wall4,floorPT,intWall,celinig1]
#     #surfaces2=[wall1,wall2,wall3,wall4,roofP1,intWall,celinig2]
    
#     #zone1.zoneParameter(surfaces1,envelopes[archId])
#     #zone2.zoneParameter(surfaces2,envelopes[archId])
    
#     wall1=[[0,0,0],[1,0,0],[1,0,9.5],[0,0,9.5]]
#     wall2=[[0,0,0],[0,0,9.5],[0,1,9.5],[0,1,0]]
#     wall3=[[0,1,0],[0,1,9.5],[1,1,9.5],[1,1,0]]
#     wall4=[[1,0,0],[1,1,0],[1,1,9.5],[1,0,9.5]]
    
#     floorPT = [[0,0,0],[0,1,0],[1,1,0],[1,0,0]]
#     roofP1 = [[0,0,9.5],[1,0,9.5],[1,1,9.5],[0,1,9.5]]
    
#     surfaces=[wall1,wall2,wall3,wall4,floorPT,roofP1]
    
    
#     edifici = {}
    
#     for i in range(1):
#         edifici[str(i)] = Building(buildingName, surfaces,envelopes,archId,sched2,'Office',SolarGains,T_ext,dT_er)
#         #edifici[str(i)].load_calculation(SolarGains,dT_er)
  
#     for t in time:

#         T_e = T_ext.iloc[t]
#         sched = sched_zone.sched_df.iloc[t]
#         Solar_gain = SolarGains.iloc[t]
#         print(t)
        
#         for i in range(1):
#             #edifici['0'].load_calculation(t,Solar_gain,dT_er)
#             edifici['0'].zones['Zone'].solve(t,T_e)
