# 1. Load a ROS2 map defined by png/pgm and yaml file
# 2. User edit the map and define track surface
#     a) user can use left, middle, right button on mouse to mark pixel
#        as occupied, free, unknown
#     b) User then click botton: label track surface and click inside the track surface
#            , which must be a closed area, this will take some time
#     c) TODO add ability to save modified map
#     d) User then draw a reference path
# 3. Trajectory optimization

# TODO: trajectory violates bonds, likely problem with trackBoundary

import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import cos,sin,pi,atan2,radians,degrees,tan
from scipy.interpolate import splprep, splev,CubicSpline,interp1d

from .common import *
from .MapProcessor import MapProcessor
from .CurvilinearTrack import CurvilinearTrack

class MapTrack(CurvilinearTrack):
    def __init__(self, mymap):
        CurvilinearTrack.__init__(self)
        self.map = mymap
        return

    def buildTrackFromMap(self,  show_track = True):
        mymap = self.map
        print(f"DEBUG: Number of raw reference pixels from map: {len(mymap.reference_path_pixels)}") # <-- ADD THIS

        # higher the number, smoother the initial reference
        tol = 0.1

        # convert pixels to coordinates
        # ... inside buildTrackFromMap ...
        ref_pixels = np.array(mymap.reference_path_pixels)
        print(f"DEBUG: Number of raw reference pixels: {len(ref_pixels)}") # Add this too for comparison
        mask = np.where(np.abs(np.diff(ref_pixels[:,0])) + np.abs(np.diff(ref_pixels[:,1])) > 0)
        ref_pixels = np.r_[ref_pixels[mask], ref_pixels[[-1], :]]
        print(f"DEBUG: Number of reference pixels after filtering: {len(ref_pixels)}") # <-- ADD THIS
        # ... rest of the function

        # shape: (N, 2)
        r = ref_path = np.array([ mymap.pixel2m(coord) for coord in ref_pixels])


        assert (len(r.shape) == 2)
        assert (r.shape[1] == 2)
        n = r.shape[0]
        xx = r[:,0]
        yy = r[:,1]

        s = 0
        ss = [s]
        for i in range(n):
            s += ((xx[(i+1)%n]-xx[i])**2 +(yy[(i+1)%n]-yy[i])**2 )**0.5
            ss.append(s)
        raceline_len_m = s
        r = np.vstack([r,r[-1]])

        plt.plot(r[:,0], r[:,1])
        plt.show()
        s = n * tol**2

        raceline_s, u = splprep(r.T, u=ss,s=s,per=1)

        # let raceline curve be r(u)
        # dr = r'(u), parameterized with xx/u
        dr = np.array(splev(ss,raceline_s,der=1))
        # ddr = r''(u)
        ddr = np.array(splev(ss,raceline_s,der=2))
        _norm = lambda x:np.linalg.norm(x,axis=0)
        # radius of curvature can be calculated as R = |y'|^3/sqrt(|y'|^2*|y''|^2-(y'*y'')^2)
        curvature = 1.0/(_norm(dr)**3/(_norm(dr)**2*_norm(ddr)**2 - np.sum(dr*ddr,axis=0)**2)**0.5)
        curvature_fun, u = splprep(curvature.reshape(1,-1), u=ss,s=0,per=1)
        # n*2
        print(f"DEBUG: self.discretized_raceline_len = {self.discretized_raceline_len}") # <-- ADD THIS
        # --- START OF REPLACEMENT ---

        # Evaluate spline at 'discretized_raceline_len' points
        ss = np.linspace(0, raceline_len_m, self.discretized_raceline_len)

        # Evaluate position (r_vec) and derivative (dr_vec)
        # splev returns list [x_coords, y_coords], so np.array converts to shape (2, N)
        r_vec_raw = np.array(splev(ss, raceline_s, der=0))
        dr_vec_raw = np.array(splev(ss, raceline_s, der=1))

        # Transpose r_vec_raw to get the desired (N, 2) shape for points
        r_vec = r_vec_raw.T
        print(f"DEBUG: Shape of r_vec (centerline points): {r_vec.shape}") # Should be (N, 2) e.g., (1024, 2)

        # Calculate heading angles (phi) from the derivative vector (shape (2, N))
        phi = np.arctan2(dr_vec_raw[1,:], dr_vec_raw[0,:]) # Shape (N,)

        # Calculate lateral vectors (perpendicular to heading)
        lateral = np.vstack([np.cos(phi + np.pi / 2), np.sin(phi + np.pi / 2)]).T # Shape (N, 2)

        # Get boundary points and distances using the (N, 2) r_vec
        # NOTE: getBoundary expects r_vec with shape (N, 2) according to its usage
        upper, lower, left_boundary, right_boundary = self.getBoundary(r_vec, lateral) # Pass r_vec directly

        # --- Assign attributes with correct shapes ---
        # discretized raceline center progress (independent variable)
        self.ss = ss # Shape (N,)

        # centerline positions, correspond to self.ss
        self.r = self.r_vec = r_vec # Shape (N, 2)
        self.raceline_points = r_vec # Shape (N, 2) <-- CORRECTED ASSIGNMENT

        # heading angles
        self.raceline_headings = phi # Shape (N,)

        # Spline function for curvature (remains unchanged)
        self.curvature_fun = curvature_fun

        # Total raceline length (scalar)
        self.raceline_len_m = raceline_len_m

        # Spline function for centerline (remains unchanged)
        self.raceline_s = raceline_s

        # Discretized position vectors for left/right boundary points
        self.raceline_left_boundary_points = upper # Shape (N, 2)
        self.raceline_right_boundary_points = lower # Shape (N, 2)

        # Track extents (remains unchanged)
        self.x_min = np.min( np.hstack([upper[:,0],lower[:,0]]) ) - 0.1
        self.x_max = np.max( np.hstack([upper[:,0],lower[:,0]]) ) + 0.1
        self.y_min = np.min( np.hstack([upper[:,1],lower[:,1]]) ) - 0.1
        self.y_max = np.max( np.hstack([upper[:,1],lower[:,1]]) ) + 0.1

        # Boundary distances from centerline
        self.raceline_left_boundary = left_boundary # Shape (N,)
        self.raceline_right_boundary = right_boundary # Shape (N,)

        # Spline functions for boundary distances (remains unchanged)
        self.raceline_left_boundary_fun, _ = splprep(self.raceline_left_boundary.reshape(1,-1), u=ss,s=0,per=1)
        self.raceline_right_boundary_fun, _ = splprep(self.raceline_right_boundary.reshape(1,-1), u=ss,s=0,per=1)

        # --- END OF REPLACEMENT ---
        '''
        ss = np.linspace(0,raceline_len_m,self.discretized_raceline_len)
        r_vec = np.array(splev(ss,raceline_s,der=0))
        dr_vec = np.array(splev(ss,raceline_s,der=1))
        phi = np.arctan2(dr_vec[1,:], dr_vec[0,:])
        lateral = np.vstack([np.cos(phi+np.pi/2), np.sin(phi+np.pi/2)]).T

        # N*2
        upper, lower, left_boundary, right_boundary = self.getBoundary(r_vec.T, lateral)

        # discretized raceline center progress
        self.ss = ss
        # centerline positions, correspond to self.ss (n*2)
        self.r = self.r_vec =  r_vec.T
        self.raceline_points = r_vec
        self.curvature_fun = curvature_fun
        # heading
        self.raceline_headings = phi
        # raceline length in meters
        self.raceline_len_m = raceline_len_m
        # spline for centerline(raceline) defined on [0,raceline_len_m]
        self.raceline_s = raceline_s
        # discretized position vector for left/right boundary
        self.raceline_left_boundary_points = upper
        self.raceline_right_boundary_points = lower
        '''
        self.x_min = np.min( np.hstack([upper[:,0],lower[:,0]]) ) - 0.1
        self.x_max = np.max( np.hstack([upper[:,0],lower[:,0]]) ) + 0.1
        self.y_min = np.min( np.hstack([upper[:,1],lower[:,1]]) ) - 0.1
        self.y_max = np.max( np.hstack([upper[:,1],lower[:,1]]) ) + 0.1

        # left boundary distance to centerline
        self.raceline_left_boundary = left_boundary
        self.raceline_right_boundary = right_boundary

        self.raceline_left_boundary_fun, _ = splprep(self.raceline_left_boundary.reshape(1,-1), u=ss,s=0,per=1) 
        self.raceline_right_boundary_fun, _ = splprep(self.raceline_right_boundary.reshape(1,-1), u=ss,s=0,per=1) 
        #self.discretized_raceline = np.vstack([self.raceline_points,self.raceline_headings, self.raceline_left_boundary, self.raceline_right_boundary]).T
        # Combine centerline points (N, 2) and 1D arrays (N,) using column_stack
        self.discretized_raceline = np.column_stack([
            self.raceline_points,           # Shape (N, 2)
            self.raceline_headings,         # Shape (N,)
            self.raceline_left_boundary,    # Shape (N,)
            self.raceline_right_boundary    # Shape (N,)
        ]) # Result shape should be (N, 5)
        print(f"DEBUG: Shape of self.discretized_raceline: {self.discretized_raceline.shape}")
        #self.upper_fun = self.buildSpline(upper)
        #self.lower_fun = self.buildSpline(lower)

        self.raceline_speed_s = interp1d(ss,np.ones_like(ss),kind='cubic')

        if show_track:
            plt.plot(upper[:,0],upper[:,1], '*')
            plt.plot(lower[:,0],lower[:,1], 'o')
            plt.plot(r_vec[0,:],r_vec[1,:],'-')
            plt.show()
        return

    def preciseTrackBoundary(self,coord,heading):
        mymap = self.map
        x = coord[0]
        y = coord[1]
        dxx = self.r[:,0]-x
        dyy = self.r[:,1]-y
        index = np.argmin(dxx**2+dyy**2)

        # find offset
        # positive offset means car is to the left of the trajectory(need to turn right)
        dr = self.r[(index+2)%self.discretized_raceline_len] - self.r[index]
        track_to_car = (x-self.r[index,0], y-self.r[index,1])
        offset = np.cross(dr/np.linalg.norm(dr),track_to_car).item()

        phi = np.arctan2(dr[1], dr[0])
        lateral = np.vstack([np.cos(phi+np.pi/2), np.sin(phi+np.pi/2)]).T[0]

        # trick to avoid left=right=0 when test point is just outside the track
        left = 0.05
        test_pt = tuple(coord + lateral * left)
        valid = mymap.isValidTrack(test_pt)
        if valid:
            while valid:
                left += 0.01
                test_pt = tuple(coord + lateral * left)
                valid = mymap.isValidTrack(test_pt)
        else:
            while not valid:
                left -= 0.01
                test_pt = tuple(coord + lateral * left)
                valid = mymap.isValidTrack(test_pt)

        right = 0.05
        test_pt = tuple(coord - lateral * right)
        valid = mymap.isValidTrack(test_pt)
        if (valid):
            while valid:
                right += 0.01
                test_pt = tuple(coord - lateral * right)
                valid = mymap.isValidTrack(test_pt)
        else:
            while not valid:
                right -= 0.01
                test_pt = tuple(coord - lateral * right)
                valid = mymap.isValidTrack(test_pt)
        if (left < 1e-3 and right < 1e-3):
            self.print_warning('left and right boundary margin are both zero')

        return (left,right)

    def getBoundary(self, r_vec, lateral):
        mymap = self.map
        #upper = r_vec.T + lateral * self.width/2
        left_boundary = []
        upper = []
        for i in range(self.discretized_raceline_len):
            coord = r_vec[i]
            half_width = 0
            test_pt = tuple(coord + lateral[i] * half_width)
            valid = mymap.isValidTrack(test_pt)
            while valid:
                half_width += 0.01
                test_pt = tuple(coord + lateral[i] * half_width)
                valid = mymap.isValidTrack(test_pt)
            left_boundary.append(half_width)
            upper.append(test_pt)
        left_boundary = np.array(left_boundary)
        upper = np.array(upper)

        #lower = r_vec.T - lateral * self.width/2
        right_boundary = []
        lower = []
        for i in range(self.discretized_raceline_len):
            coord = r_vec[i]
            half_width = 0
            test_pt = tuple(coord - lateral[i] * half_width)
            valid = mymap.isValidTrack(test_pt)
            while valid:
                half_width += 0.01
                test_pt = tuple(coord - lateral[i] * half_width)
                valid = mymap.isValidTrack(test_pt)
            right_boundary.append(half_width)
            lower.append(test_pt)
        lower = np.array(lower)
        right_boundary = np.array(right_boundary)
        return upper, lower, left_boundary, right_boundary

if __name__ ==  '__main__':
    # load a map, edit it, and save it
    mymap = MapProcessor()
    #mymap.readPng('maps/f1tenth_maps/maps/Oschersleben_map.png')
    #mymap.readYaml('maps/f1tenth_maps/maps/Oschersleben_map.yaml')
    name = 'berlin'
    # for reading png
    #mymap.readPng(f'maps/f1tenth_maps/maps/{name}.png')
    # some maps are not encoded with 0-254 but with 0.0-1.0, convert the range
    #mymap.data *= 254
    # for reading pgm
    mymap.readPgm(f'maps/f1tenth_maps/thisisthefr.pgm')
    mymap.readYaml(f'maps/f1tenth_maps/thisisthefr.yaml')
    mymap.interactiveMapEdit()
    with open('./map.p', 'wb') as f:
        pickle.dump(mymap,f)

    '''
    # load saved map
    with open('./map.p', 'rb') as f:
        mymap = pickle.load(f)
    '''

    track = MapTrack(mymap)
    track.buildTrackFromMap()
    # offset: safety margin to track boundary
    # N: how many reference points to use when constructing the track
    track.optimizePath(offset=0.5, N = 200, visualize=False, save_gif=False)

    # show left/right boundary
    phi = track.raceline_headings
    lateral = np.vstack([np.cos(phi+np.pi/2), np.sin(phi+np.pi/2)]).T
    upper, lower, left_boundary, right_boundary = track.getBoundary(track.r, lateral)
    # display optimized trajectory and track boundaries
    #mymap.displayPath([track.r, upper, lower])
    # display optimized trajectory only
    mymap.displayPath([track.r])


