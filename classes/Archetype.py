# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 12:02:19 2021

@author: fabix
"""
import pandas as pd
from pandas import DataFrame
import os
import csv
import pvlib
import numpy as np
import math


class Archetypes(object):
    
    def __init__(self,i,istat_df):
                
        ''' 
        Associazione archetipo (archID) ad edificio ISTAT 
        '''
        
        building_age = istat_df["q_2_4_ric"] # Build_Year
        external_wall_typology = istat_df["q_2_25"] # ExtWall_Type
        window_typology  = istat_df["win_id"] # win type  
        building_typology = istat_df["q_2_1"] # Build type
        floor_surface = istat_df["q_2_7_class"] # build surf
        apartment_typology = istat_df["q_2_2"] # floor type
        codId = istat_df["id"] # n_building
        North = istat_df["q_2_29_1"] # Orient N
        East = istat_df["q_2_29_2"] # Orient E
        South = istat_df["q_2_29_3"] # Orient S
        West = istat_df["q_2_29_4"] # Orient W
        no_oriented_surf = istat_df["q_2_29_9"] # Orient 0
        env_insulation = istat_df["q_2_26"] # Insulation
        env_insulation_new = istat_df["q_3_19_7"]  # Insulation ref    
        
        n_fin_legno = 0 if math.isnan(istat_df["q_2_11_class"][i]) else istat_df["q_2_11_class"][i]
        n_fin_met = 0 if math.isnan(istat_df["q_2_15_class"][i]) else istat_df["q_2_15_class"][i]
        n_fin_pvc = 0 if math.isnan(istat_df["q_2_19_class"][i]) else istat_df["q_2_19_class"][i]
        
        n_fin_legno = 10 if n_fin_legno > 10 else n_fin_legno
        n_fin_met = 10 if n_fin_met > 10 else n_fin_met 
        
        n_finestre =  n_fin_legno + n_fin_met + n_fin_pvc      
        n_finestre = 2 if n_finestre == 0 else n_finestre
        self.area_fin = n_finestre * 1.2*1.4
        
        building_age_class = building_age[i]
        external_wall_material= external_wall_typology[i]
        window_frame_material= window_typology[i]
        floor_surface_class = floor_surface[i]
        building_typology_class = building_typology[i]
        apartment_floor_typology = apartment_typology[i]
        self.name = codId[i]
        '''
        env_insulation: presenza di isolamento termico
        env_insulation_new: intervento di riqualificazione effettuato negli ultimi 5 anni (2008-13)
        building_age: classe di età dell'edificio (stato originale)
        '''
        insulation_layer = env_insulation[i]
        new_insulation_layer = env_insulation_new[i]
        
        if no_oriented_surf[i] == 1: 
            self.orientation = np.array([0, 1, 0, 1])
            self.orientation_txt = 'EW'
        else:  
            self.orientation = np.array([North[i], East[i], South[i], West[i]])
            self.orientation_txt = {0:'',1:'N'}[North[i]]+{0:'',1:'E'}[East[i]]+{0:'',1:'S'}[South[i]] +{0:'',1:'W'}[West[i]]
        # self.orientation = np.array([1, 1, 1, 1])
        
        if building_age_class== 9 or building_age_class== 8: 
            self.building_age_class = "pre-1900"
        elif building_age_class== 7: self.building_age_class = "1900-1949"
        elif building_age_class== 6: self.building_age_class = "1950-1959"
        elif building_age_class== 5: self.building_age_class = "1960-1969"
        elif building_age_class== 4: self.building_age_class = "1970-1979"
        elif building_age_class== 3: self.building_age_class = "1980-1989"
        elif building_age_class== 2: self.building_age_class = "1990-1999"             
        elif building_age_class== 1: self.building_age_class = "post-2000"
        else: exit(code= "No building age assignment") 

        
        if external_wall_material==1: self.external_wall_material= "Concrete"
        elif external_wall_material==2: self.external_wall_material= "Stone" 
        elif external_wall_material==3: self.external_wall_material= "Masonary" 
        elif external_wall_material==4: self.external_wall_material= "Wood" 
        elif external_wall_material==9: self.external_wall_material= "None" 
        else: print("No building external wall material assignment") 

        
        if window_frame_material== 1: self.window_frame_material= "WoodFrame"
        elif window_frame_material== 2: self.window_frame_material= "MetalFrame"
        else: print("No building window frame material assignment") 
     
        if   floor_surface_class==1: self.floorArea = 15; # [mq]
        elif floor_surface_class==2: self.floorArea = 30 # [mq]
        elif floor_surface_class==3: self.floorArea = 50 # [mq]
        elif floor_surface_class==4: self.floorArea = 75 # [mq]
        elif floor_surface_class==5: self.floorArea = 105 # [mq]
        elif floor_surface_class==6: self.floorArea = 135 # [mq]
        elif floor_surface_class==7: self.floorArea = 165 # [mq]
        else: print("No building floor surface assignment") 
      
        if building_typology_class== 1: self.building_typology_class= "SFH"
        elif building_typology_class== 2: self.building_typology_class= "MFH"
        elif building_typology_class== 3: self.building_typology_class= "AB_LowDensity"
        elif building_typology_class== 4: self.building_typology_class= "AB_MediumDensity"
        elif building_typology_class== 5: self.building_typology_class= "AB_HighDensity"
        else: print("No building typology assignment")       
           
        if apartment_floor_typology== 1: self.apartment_floor_typology= "BasementFloor"
        elif apartment_floor_typology== 2: self.apartment_floor_typology= "GroundFloor"
        elif apartment_floor_typology== 3: self.apartment_floor_typology= "IntermediateFloor"
        elif apartment_floor_typology== 4: self.apartment_floor_typology= "LastFloor"
        else: self.apartment_floor_typology= ""
                                        
        self.assignArchetype()
        
        self.surfaces = dict()
        self.createSurfaces()
        
        self.assignInsulation(insulation_layer,new_insulation_layer)
        
        

    def assignArchetype(self):

        if self.building_age_class== "pre-1900":         
            if self.external_wall_material== "Stone" or self.external_wall_material== "Wood" or self.external_wall_material== "None":    
                 if self.window_frame_material== "MetalFrame":
                     self.archID = 1
                 elif self.window_frame_material== "WoodFrame":
                     self.archID = 4              
            if self.external_wall_material== "Concrete":          
                 if self.window_frame_material== "MetalFrame":
                     self.archID = 2
                 elif self.window_frame_material== "WoodFrame":
                     self.archID = 5               
            if self.external_wall_material== "Masonary":           
                 if self.window_frame_material== "MetalFrame":
                     self.archID = 3
                 elif self.window_frame_material== "WoodFrame":
                     self.archID = 6          
     
        if self.building_age_class== "1900-1949": 
            if self.external_wall_material== "Masonary" or self.external_wall_material== "Wood" or self.external_wall_material== "None":           
                if self.window_frame_material== "MetalFrame":
                    self.archID = 7
                elif self.window_frame_material== "WoodFrame":
                    self.archID = 10               
            if self.external_wall_material== "Concrete":  
                if self.window_frame_material== "MetalFrame":
                    self.archID = 8
                elif self.window_frame_material== "WoodFrame":
                  self.archID = 11  
            if self.external_wall_material== "Stone":
                if self.window_frame_material== "MetalFrame":
                    self.archID = 9
                elif self.window_frame_material== "WoodFrame":
                    self.archID = 12
 
        if self.building_age_class== "1950-1959": 
            if self.external_wall_material== "Masonary" or self.external_wall_material== "Wood" or self.external_wall_material== "None":     
                if self.window_frame_material== "MetalFrame":
                    self.archID = 16
                elif self.window_frame_material== "WoodFrame":
                    self.archID = 13  
            if self.external_wall_material== "Concrete":          
                if self.window_frame_material== "MetalFrame":
                    self.archID = 17
                elif self.window_frame_material== "WoodFrame":
                    self.archID = 14              
            if self.external_wall_material== "Stone":          
                if self.window_frame_material== "MetalFrame":
                    self.archID = 18
                elif self.window_frame_material== "WoodFrame":
                    self.archID = 15
               
        if self.building_age_class== "1960-1969": 
            if self.external_wall_material== "Masonary" or self.external_wall_material== "Wood" or self.external_wall_material== "None":          
                if self.window_frame_material== "WoodFrame":
                    self.archID = 19
                elif self.window_frame_material== "MetalFrame":
                    self.archID = 22              
            if self.external_wall_material== "Concrete":           
                if self.window_frame_material== "WoodFrame":
                    self.archID = 20
                elif self.window_frame_material== "MetalFrame":
                    self.archID = 23              
            if self.external_wall_material== "Stone":           
                if self.window_frame_material== "WoodFrame":
                    self.archID = 21
                elif self.window_frame_material== "MetalFrame":
                    self.archID = 24
    
        if self.building_age_class== "1970-1979": 
            if self.external_wall_material== "Masonary" or self.external_wall_material== "Wood" or self.external_wall_material== "None":  
                if self.window_frame_material== "MetalFrame":
                    self.archID = 25
                elif self.window_frame_material== "WoodFrame":
                    self.archID = 28
            if self.external_wall_material== "Concrete":           
                if self.window_frame_material== "MetalFrame":
                    self.archID = 26
                elif self.window_frame_material== "WoodFrame":
                    self.archID = 29               
            if self.external_wall_material== "Stone":          
                if self.window_frame_material== "MetalFrame":
                    self.archID = 27
                elif self.window_frame_material== "WoodFrame":
                    self.archID = 30
   
        if self.building_age_class== "1980-1989":  
            if self.external_wall_material== "Masonary" or self.external_wall_material== "Wood" or self.external_wall_material== "None":           
                if self.window_frame_material== "MetalFrame":
                    self.archID = 31
                elif self.window_frame_material== "WoodFrame":
                    self.archID = 34               
            if self.external_wall_material== "Concrete":           
                if self.window_frame_material== "MetalFrame":
                    self.archID = 32
                elif self.window_frame_material== "WoodFrame":
                    self.archID = 35              
            if self.external_wall_material== "Stone":          
                if self.window_frame_material== "MetalFrame":
                    self.archID = 33
                elif self.window_frame_material== "WoodFrame":
                    self.archID = 36
   
        if self.building_age_class== "1990-1999":       
            if self.external_wall_material== "Masonary" or self.external_wall_material== "Wood" or self.external_wall_material== "None":          
                if self.window_frame_material== "MetalFrame":
                    self.archID = 37
                elif self.window_frame_material== "WoodFrame":
                    self.archID = 40              
            if self.external_wall_material== "Concrete":           
                if self.window_frame_material== "MetalFrame":
                    self.archID = 38
                elif self.window_frame_material== "WoodFrame":
                    self.archID = 41               
            if self.external_wall_material== "Stone":           
                if self.window_frame_material== "MetalFrame":
                    self.archID = 39
                elif self.window_frame_material== "WoodFrame":
                    self.archID = 42                     
    
        if self.building_age_class== "post-2000":        
            if self.external_wall_material== "Masonary" or self.external_wall_material== "Wood" or self.external_wall_material== "None":           
                if self.window_frame_material== "WoodFrame":
                    self.archID = 43
                elif self.window_frame_material== "MetalFrame":
                    self.archID = 46               
            if self.external_wall_material== "Concrete":          
                if self.window_frame_material== "WoodFrame":
                    self.archID = 44
                elif self.window_frame_material== "MetalFrame":
                    self.archID = 47               
            if self.external_wall_material== "Stone":           
                if self.window_frame_material== "WoodFrame":
                    self.archID = 45
                elif self.window_frame_material== "MetalFrame":
                    self.archID = 48
        
    
    def orientedSurfaces(self, N = 1):
        
        # Parametri
        h = 3.3  # altezza lorda singolo piano
        spessore_muro_medio = 0.8 # Doppio, per tenere conto dell'area lorda [m] 
        moltiplicatore_area_lorda = 1.0 
        
        H = N*h  # altezza lorda totale
        surf_groundfloor = self.floorArea*moltiplicatore_area_lorda/N
        surf_rooftop     = self.floorArea*moltiplicatore_area_lorda/N
        
        L = np.sqrt(surf_groundfloor) + spessore_muro_medio # [m]
        
        surf_to_north = (L*H)*self.orientation[0] 
        surf_to_east  = (L*H)*self.orientation[1]
        surf_to_south = (L*H)*self.orientation[2]
        surf_to_west  = (L*H)*self.orientation[3] 
        
        wwr = self.area_fin/(surf_to_north + surf_to_east + surf_to_south + surf_to_west)
        if wwr < 0.05:
            wwr = 0.05
        elif wwr > 0.25:
            wwr = 0.25 
        
        # wwr = [0.125,0.125,0.125,0.125] ## to be changed <------------------------------------------------
        
        opaque_surf_to_north = surf_to_north*(1-wwr)
        opaque_surf_to_east  = surf_to_east*(1-wwr)
        opaque_surf_to_south = surf_to_south*(1-wwr)
        opaque_surf_to_west  = surf_to_west*(1-wwr)
        
        self.surfaces = [{
                        'inclination': 90.0, # vertical surface (wall)
                        'area': surf_to_north,
                        'area_adjacent': 0,
                        'shading': 'none',
                        'glazing_ratio': wwr,
                        'orientation': 180.0, # north (azimuth = 180°)
                        'type': 'ext_wall',
                        'area_opaque': opaque_surf_to_north, #<--------------------------------------------
                        'name': 'ext_wall_north'},
                        {
                        'inclination': 90.0, # vertical surface (wall)
                        'area': surf_to_east,
                        'area_adjacent': 0,
                        'shading': 'none',
                        'glazing_ratio':  wwr,
                        'orientation': 90.0, # west (azimuth = 90°)
                        'type': 'ext_wall',
                        'area_opaque': opaque_surf_to_east,
                        'name': 'ext_wall_east'},   
                        {
                        'inclination': 90.0, # vertical surface (wall)
                        'area': surf_to_south,
                        'area_adjacent': 0,
                        'shading': 'none',
                        'glazing_ratio':  wwr,
                        'orientation': 0.0, # south (azimuth = 0°)
                        'type': 'ext_wall',
                        'area_opaque': opaque_surf_to_south,
                        'name': 'ext_wall_south'},
                        {
                        'inclination': 90.0, # vertical surface (wall)
                        'area': surf_to_west,
                        'area_adjacent': 0,
                        'shading': 'none',
                        'glazing_ratio':  wwr,
                        'orientation': 270.0, # south (azimuth = 0°)
                        'type': 'ext_wall',
                        'area_opaque': opaque_surf_to_west,
                        'name': 'ext_wall_west'}
                        ]
        
        if self.building_typology_class in ["SFH","MFH"]:
            self.surfaces.append({
                            'inclination': 0, # vertical surface (wall)
                            'area': surf_rooftop,
                            'area_adjacent': 0,
                            'shading': 'none',
                            'glazing_ratio': 0.0, 
                            'orientation': 0.0, # north (azimuth = 180°)
                            'type': 'roof',
                            'area_opaque': surf_rooftop,
                            'name': 'roof'},)
            self.surfaces.append({
                            'inclination': 180, # vertical surface (wall)
                            'area': surf_groundfloor,
                            'area_adjacent': 0,
                            'shading': 'none',
                            'glazing_ratio': 0.0, 
                            'orientation': 0.0, # north (azimuth = 180°)
                            'type': 'floor',
                            'area_opaque': surf_groundfloor,
                            'name': 'floor'},)

        if "AB" in self.building_typology_class:
            if self.apartment_floor_typology in ["LastFloor"]:
                self.surfaces.append({
                                'inclination': 0, # vertical surface (wall)
                                'area': surf_rooftop,
                                'area_adjacent': 0,
                                'shading': 'none',
                                'glazing_ratio': 0.0, 
                                'orientation': 0.0, # north (azimuth = 180°)
                                'type': 'roof',
                                'area_opaque': surf_rooftop,
                                'name': 'roof'},)
            if self.apartment_floor_typology in ["BasementFloor","GroundFloor"]:
                self.surfaces.append({
                                'inclination': 180, # vertical surface (wall)
                                'area': surf_groundfloor,
                                'area_adjacent': 0,
                                'shading': 'none',
                                'glazing_ratio': 0.0, 
                                'orientation': 0.0, # north (azimuth = 180°)
                                'type': 'floor',
                                'area_opaque': surf_groundfloor,
                                'name': 'floor'},)
        

        # come inserire adiacent walls etc?
        
        self.dwelling_height = H
        self.nFloors         = N                  # no. of floors
        # self.floorArea   = self.floorArea/N     # footprint   
        self.volume   = self.floorArea*H     # footprint
    
    
            
    def createSurfaces(self):
           

        if self.building_typology_class == "SFH":  
            '''
            SINGLE FAMILY HOUSE
            (hyp: 2 storeys, isolated, 1 groundfloor, 1 internalfloor)
            '''
            N = 2           
            self.orientedSurfaces(N)

            self.AdSurf= [['IntCeiling', self.floorArea/N],
                          ['IntFloor', self.floorArea/N],
                          ['IntWall', 2*self.floorArea]]
            self.archType = 'SFH'
              
    
        elif self.building_typology_class == "MFH": # Casa Plurifamiliare/Villetta a schiera        
            '''
            MULTI FAMILY HOUSE - TERRACED HOUSE
            (hyp: 2 storeys, 1 adjacent building, 1 groundfloor, 1 internalfloor)
            '''
            N = 2
            self.orientedSurfaces(N)
            
            self.AdSurf = [['IntCeiling', self.floorArea/N],
                           ['IntFloor', self.floorArea/N],
                           ['IntWall', 2*self.floorArea]]
            self.archType = 'MFH or TH'
            
        
        else: 
             '''
             APARTMENT - GROUNDFLOOR
             (hyp: 1 storeys, 1 groundfloor, 0 internalfloor, 1 ceiling)
             '''
             N = 1
             
             self.orientedSurfaces(N)
    
             if self.apartment_floor_typology == "BasementFloor" or self.apartment_floor_typology == "GroundFloor":   
 
                 self.AdSurf = [['IntCeiling', self.floorArea],
                                ['IntFloor', 0],
                                ['IntWall', 2*self.floorArea]]
                 self.archType = 'AB_GroundFloor'
             
      
             elif  self.apartment_floor_typology == "IntermediateFloor":                   
                 '''
                 APARTMENT - INTERMEDIATE FLOOR
                 (hyp: 1 storeys, 0 groundfloor, 1 internalfloor, 1 ceiling)
                 '''                
                 self.AdSurf = [['IntCeiling', self.floorArea],
                                ['IntFloor', self.floorArea],
                                ['IntWall', 2*self.floorArea]]
                 self.archType = 'AB_InterFloor'
                 
             
             elif self.apartment_floor_typology == "LastFloor":                                      
                 '''
                 APARTMENT - LAself.floorAreaT FLOOR
                 (hyp: 1 storeys, 0 groundfloor, 1 internalfloor, 0 ceiling)
                 '''
                 
                 self.AdSurf = [['IntCeiling', 0],
                                ['IntFloor', self.floorArea],
                                ['IntWall', 2*self.floorArea]]
                 self.archType = 'AB_LastFloor'
         
                 
        
            
       
    def assignInsulation(self,insulation_layer,new_insulation_layer):
        
        if self.building_age_class== "pre-1900":         
            if insulation_layer == 1 and new_insulation_layer == 1:
                add_insulation_layer = "HighInsulationLevel"
            elif insulation_layer == 1 and new_insulation_layer == 0: 
                add_insulation_layer = "LowInsulationLevel"
            else: add_insulation_layer = "None"  
            self.insulator_layer = add_insulation_layer                    
    
        elif self.building_age_class== "1900-1949": 
            if insulation_layer == 1 and new_insulation_layer == 1:
                add_insulation_layer = "HighInsulationLevel"
            elif insulation_layer == 1 and new_insulation_layer == 0: 
                add_insulation_layer = "LowInsulationLevel"
            else: add_insulation_layer = "None" 
            self.insulator_layer = add_insulation_layer
        
        elif self.building_age_class== "1950-1959":             
            if insulation_layer == 1 and new_insulation_layer == 1:
                add_insulation_layer = "HighInsulationLevel"
            elif insulation_layer == 1 and new_insulation_layer == 0: 
                add_insulation_layer = "LowInsulationLevel"
            else: add_insulation_layer = "None" 
            self.insulator_layer = add_insulation_layer
                 
        elif self.building_age_class== "1960-1969": 
            if insulation_layer == 1 and new_insulation_layer == 1:
                add_insulation_layer = "HighInsulationLevel"
            elif insulation_layer == 1 and new_insulation_layer == 0: 
                add_insulation_layer = "LowInsulationLevel"
            else: add_insulation_layer = "None" 
            self.insulator_layer = add_insulation_layer
    
        elif self.building_age_class== "1970-1979": 
            if insulation_layer == 1 and new_insulation_layer == 1:
                add_insulation_layer = "HighInsulationLevel"
            elif insulation_layer == 1 and new_insulation_layer == 0: 
                add_insulation_layer = "LowInsulationLevel"
            else: add_insulation_layer = "None" 
            self.insulator_layer = add_insulation_layer
    
        elif self.building_age_class== "1980-1989":  
            if insulation_layer == 1 and new_insulation_layer == 1:
                add_insulation_layer = "HighInsulationLevel"
            elif insulation_layer == 1 and new_insulation_layer == 0: 
                add_insulation_layer = "LowInsulationLevel"
            else: add_insulation_layer = "None"  
            self.insulator_layer = add_insulation_layer           
    
        elif self.building_age_class== "1990-1999":                  
            if insulation_layer == 1 and new_insulation_layer == 1:
                add_insulation_layer = "HighInsulationLevel"
            elif insulation_layer == 1 and new_insulation_layer == 0: 
                add_insulation_layer = "LowInsulationLevel"
            else: add_insulation_layer = "None" 
            self.insulator_layer = add_insulation_layer
              
        elif self.building_age_class== "post-2000":        
            if insulation_layer == 1 and new_insulation_layer == 1:
                add_insulation_layer = "HighInsulationLevel"
            elif insulation_layer == 1 and new_insulation_layer == 0: 
                add_insulation_layer = "LowInsulationLevel"
            else: add_insulation_layer = "None" 
            self.insulator_layer = add_insulation_layer
        


