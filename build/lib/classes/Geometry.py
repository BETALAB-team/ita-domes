'''IMPORTING MODULES'''

import os
import numpy as np
# import pyclipper as pc
from classes.Envelope import loadEnvelopes

#%% Useful functions

def unit_normal(a, b, c):
    
    '''
    This function starting from three points defines the normal vector of the plane
    '''
    
    

    #unit normal vector of plane defined by points a, b, and c
    x = np.linalg.det([[1,a[1],a[2]],
         [1,b[1],b[2]],
         [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
         [b[0],1,b[2]],
         [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
         [b[0],b[1],1],
         [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    if np.all(a == b) or np.all(a == c) or np.all(b == c):
        return (0,0,0)
    else:
        return (x/magnitude, y/magnitude, z/magnitude)


def poly_area(poly):
    
    '''
    From a list of points calculates the area of the polygon
    '''
    #area of polygon poly
    if len(poly) < 3:                                                          # Not a plane - no area
        print('number of vertices lower than 3, the area will be zero')
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)


def check_complanarity(vertListTot,precision=1):
    
    '''
    checks the complanarity of a list of points
    '''
    flag = True
    for i in range(len(vertListTot)-3):
        vertList = vertListTot[i:(i+4)]
        a1 = vertList[1][0] - vertList[0][0]
        b1 = vertList[1][1] - vertList[0][1]
        c1 = vertList[1][2] - vertList[0][2]
        a2 = vertList[2][0] - vertList[0][0]
        b2 = vertList[2][1] - vertList[0][1]
        c2 = vertList[2][2] - vertList[0][2]
        a = b1 * c2 - b2 * c1
        b = a2 * c1 - a1 * c2
        c = a1 * b2 - b1 * a2
        d = (- a * vertList[0][0] - b * vertList[0][1] - c * vertList[0][2])

        # equation of plane is: a*x + b*y + c*z = 0 #
        # checking if the 4th point satisfies
        # the above equation
        if not (np.abs(a * vertList[3][0] + b * vertList[3][1] + c * vertList[3][2] + d) < precision):
            flag = False

    return flag


def centroid(vertList):
    
    '''
    From a list of points calculates the centroid
    '''    
    c = np.array([0.,0,0])
    for i in vertList:
        c += np.array(i)
    return c/len(vertList)


def normalAlternative(vertList):
    
    '''
    Alternative
    This function starting from three points defines the normal vector of the plane
    '''
    c = centroid(vertList)
    crossProd=np.array([0.,0,0])
    for i in range(len(vertList)):
        a = np.array(vertList[i-1]) - c
        b = np.array(vertList[i]) - c
        crossProd += np.cross(a,b)
    if np.linalg.norm(crossProd) == 0.:
        return np.array([0.,0.,1.])
    else:
        return crossProd / np.linalg.norm(crossProd)


def project(x,proj_axis):
    
    '''
    # Project onto either the xy, yz, or xz plane. (We choose the one that avoids degenerate configurations, which is the purpose of proj_axis.)
    # In this example, we would be projecting onto the xz plane.
    '''
    return tuple(c for i, c in enumerate(x) if i != proj_axis)


def project_inv(x,proj_axis,a,v):
    
    '''
    # Returns the vector w in the walls' plane such that project(w) equals x.
    '''
    w = list(x)
    w[proj_axis:proj_axis] = [0.0]
    c = a
    for i in range(3):
        c -= w[i] * v[i]
    c /= v[proj_axis]
    w[proj_axis] = c
    return tuple(w)

#%%

class Surface:
    
    '''
    Class surface checks the complanarity and calculates the area.
    Then calculates the azimuth and tilt of the surface and set a surface
    type depending on the tilt angle
    
    __init__:
        name
        sky dome azimuth subdivision
        sky dome height subdivision
        window wall ratio
        list of the vertices
        surface type: ExtWall, Roof, GroundFloor, Ceiling, IntWall    
        
    maxHeight: no input
    
    minHeight: no input
    
    checkSurfaceCoincidence takes a second surface to check if they are co-planar and they match eachother:
        secondary surface
        
    reduceArea reduce the area of the surface:
        area to reduce
    
    '''
    # def __init__(self,name,floorArea,orientation,azSubdiv,hSubdiv,wwr,rh_gross,vertList=[[0,0,0],[0,0,0],[0,0,0]],surfType='ExtWall'):

    def __init__(self,name,floorArea,orientation,area_fin,azSubdiv,hSubdiv,rh_gross,vertList=[[0,0,0],[0,0,0],[0,0,0]],surfType='ExtWall'):
        
        '''
        input:
            list of vertices: [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],..]
            typology: one of these strings: 'ExtWall','Roof','GroundFloor'

        attributes:
            type
            area
            azimuth
            height

        methods:
            --none--

        complanarity:
            https://www.geeksforgeeks.org/program-to-check-whether-4-points-in-a-3-d-plane-are-coplanar/

        the area is calculated from:
            https://stackoverflow.com/questions/12642256/python-find-area-of-polygon-from-xyz-coordinates
        '''

        self.name = name
        self.F_r = 0.

        if not check_complanarity(vertList):
            print('surface points not planar')
            self.type = '--'
            self.area = 0
            self.normal = 0
            self.height = 0
            self.azimuth = 0
            return
        
        self.OnOff_shading = 'Off'
        self.type = surfType
        self.vertList = vertList
        self.area = poly_area(self.vertList)
        if self.area == 0 or np.isnan(self.area):
            self.area = 0.0000001
        '''
        Considering only three points in calculating the normal vector could create
        reverse orientations if the three points are in a non-convex angle of the surface

        for this reason theres an alternative way to calculate the normal,
        implemented in function: normalAlternative

        reference: https://stackoverflow.com/questions/32274127/how-to-efficiently-determine-the-normal-to-a-polygon-in-3d-space
        '''
        # self.normal = unit_normal(self.vertList[0], self.vertList[1], self.vertList[2])
        self.normal = normalAlternative(self.vertList)

        if self.normal[2] == 1:
            self.height = 0
            self.azimuth = 0
        elif self.normal[2] == -1:
            self.height= 180
            self.azimuth = 0
        else:
            self.height = 90 - np.degrees(np.arctan((self.normal[2]/(np.sqrt(self.normal[0]**2+self.normal[1]**2)))))
            if self.normal[1] == 0:
                if self.normal[0] > 0:
                    self.azimuth = -90
                elif self.normal[0] < 0:
                    self.azimuth = 90
            else:
                if self.normal[1]<0:
                    self.azimuth = np.degrees(np.arctan(self.normal[0]/self.normal[1]))
                else:
                    if self.normal[0] < 0:
                        self.azimuth = 180 + np.degrees(np.arctan(self.normal[0]/self.normal[1]))
                    else:
                        self.azimuth = -180 + np.degrees(np.arctan(self.normal[0]/self.normal[1]))

        if self.height < 40:
            self.type = 'Roof'
        if self.height > 150:
            self.type = 'GroundFloor'
        if self.type == 'ExtWall':
            self.area = self.area*rh_gross
        
        '''Azimuth and tilt approximation'''
        delta_a = 360/(2*azSubdiv)
        delta_h = 90/(2*hSubdiv)
        x = np.arange(-delta_h,90+2*delta_h,2*delta_h)
        
        for n in range(len(x)-1):
            if self.height >= x[n] and self.height < x[n+1]:
                self.height_round = int((x[n]+x[n+1])/2)
                self.F_r = (1+np.cos(np.radians(self.height_round)))/2                                                          
            elif self.height >= x[-1] and self.height < 150:
                self.height_round = 90
                self.F_r = (1+np.cos(np.radians(self.height_round)))/2
            else:
                self.height_round = 0                                          # Only to avoid errors           
                
        y = np.arange(-180-delta_a,180+2*delta_a,2*delta_a)
        for n in range(len(y)-1):
            if self.azimuth >= y[n] and self.azimuth < y[n+1]:
                self.azimuth_round = int((y[n]+y[n+1])/2)
                if self.azimuth_round == 180:
                    self.azimuth_round = -180
           
             
        if self.height_round == 0:
            self.azimuth_round = 0
            
        if self.type == 'ExtWall':
            self.centroid_coord = centroid(vertList)
            # if 135 < self.azimuth_round <= 180 or -180 <= self.azimuth_round < -135:
            #     self.wwr = wwr[0]
            # elif -135 <= self.azimuth_round <= -45:
            #     self.wwr = wwr[1]
            # elif -45 < self.azimuth_round < 45:
            #     self.wwr = wwr[2]
            # elif 45 <= self.azimuth_round <= 135:
            #     self.wwr = wwr[3]
        # else:
        #     self.wwr = 0

        # self.opaqueArea = (1-self.wwr)*self.area
        # self.glazedArea = (self.wwr)*self.area
        
        
        # MODIFICA CALC SUP OPAQUE e GLAZED
        
        
        self.floorArea = floorArea
        
        if self.area <= 0.0000001 or self.type == 'GroundFloor' or self.type == 'Roof':
            self.glazedArea = 0.0000001
        else:
            #nSurf_exp = orientation[0] + orientation[1] + orientation[2] + orientation[3]
            # self.glazedArea = round( (self.floorArea*0.125/nSurf_exp) ,2)
            # basiamo su numero finestre
            self.glazedArea = round(area_fin ,2)
        
        if self.glazedArea > self.area:
            self.glazedArea = self.area*0.9
        self.opaqueArea = round( (self.area-(self.glazedArea)) ,2)
        
        
        if self.glazedArea == 0:
            self.glazedArea = 0.0000001                                        #Avoid zero division
        if self.opaqueArea == 0:
            self.opaqueArea = 0.0000001                                        #Avoid zero division
        
                        
    def maxHeight(self):
        hmax = 0
        for vert in self.vertList:
            hmax = max(hmax,vert[2])
        return hmax


    def minHeight(self):
        hmin = 10000
        for vert in self.vertList:
            hmin = min(hmin,vert[2])
        return hmin


    def checkSurfaceCoincidence(self,otherSurface):
        flagPoints = False
        plane = self.vertList
        for i in otherSurface.vertList:
            if check_complanarity(plane + [i],precision=5):
                flagPoints = True
        flagNormal = False
        if np.linalg.norm(self.normal+otherSurface.normal)<0.2:
            flagNormal= True
        return (flagNormal and flagPoints)


    # def calculateIntersectionArea(self,otherSurface):
    #     '''
    #     reference: https://stackoverflow.com/questions/39003450/transform-3d-polygon-to-2d-perform-clipping-and-transform-back-to-3d
    #     '''
    #     a = self.normal[0]*self.vertList[0][0]+self.normal[1]*self.vertList[0][1]+self.normal[2]*self.vertList[0][2]
    #     proj_axis = max(range(3), key=lambda i: abs(self.normal[i]))
    #     projA = [project(x,proj_axis) for x in self.vertList]
    #     projB = [project(x,proj_axis) for x in otherSurface.vertList]
    #     scaledA = pc.scale_to_clipper(projA)
    #     scaledB = pc.scale_to_clipper(projB)
    #     clipper = pc.Pyclipper()
    #     clipper.AddPath(scaledA, poly_type=pc.PT_SUBJECT, closed=True)
    #     clipper.AddPath(scaledB, poly_type=pc.PT_CLIP, closed=True)
    #     intersections = clipper.Execute(pc.CT_INTERSECTION, pc.PFT_NONZERO, pc.PFT_NONZERO)
    #     intersections = [pc.scale_from_clipper(i) for i in intersections]
    #     if len(intersections)==0:
    #         return 0
    #     intersection = [project_inv(x,proj_axis,a,self.normal) for x in intersections[0]]
    #     area = poly_area(intersection)
    #     return area if area>0 else 0


    def reduceArea(self,AreaToReduce):
        if self.area - AreaToReduce > 0.0000001:
            self.area = self.area - AreaToReduce
            self.opaqueArea = (1-self.wwr)*self.area
            self.glazedArea = (self.wwr)*self.area
        else:
            self.area = 0.0000001
            self.opaqueArea = 0.0000001
            self.glazedArea = 0.0000001


    def printInfo(self):
        print('Name: '+self.name+\
              '\nArea: '+str(self.area)+\
              '\nType: '+str(self.type)+\
              '\nAzimuth: '+str(self.azimuth)+\
              '\nHeight: '+str(self.height)+\
              '\nVertices: '+str(self.vertList)+\
              '\n')


#%%

class SurfaceInternalMass():
    
    '''
    Class to define a surface for thermal capacity using area and surface type
    with a specific geometry
    
    __init__:
        name
        area
        surfaceType (IntWall, Ceiling)
    '''

    def __init__(self, name, area=0.0000001, surfType='IntWall'):
        '''
        input:
            area: area of the internal surface
            surfType: 'IntWall' or 'IntCeiling'
            adjacentZone: name of the adjacent zone ()

        attrubutes:
            area
            surfType
            adjacentZone
        '''
        self.name = name
        self.area = area
        self.type = surfType
        if self.area < 0.0000001:
            self.area = 0.0000001
        self.opaqueArea = self.area


class SurfaceInternalAdjacent(SurfaceInternalMass):
    
    '''
    Inherited from SurfaceInternalMass
    adds the adjacentZone attribute
    '''
    def __init__(self,name,area,surfType='IntCeiling',adjacentZone = None):
        super().__init__(area,surfType)
        self.adjacentZone = adjacentZone

#%%
