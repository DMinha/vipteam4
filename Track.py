# Base class for track subclasses, like CurvilinearTrack
# this class provides API for interacting with a Track object
# a Track object provides information on the trajectory and provide basic routines for drawing the track
# This is a simplified version for use by Gatech's VIP team

from common import *
import numpy as np
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
from math import radians,degrees,cos,sin,ceil,floor,atan,tan
import cv2
import os.path
import pickle
import types

class Track(PrintObject):
    def __init__(self):
        # the following variables need to be overriden in subclass initilization
        # pixels per meter
        self.resolution = None
        self.discretized_raceline_len = 1024

        # track dimension, in meters
        self.x_limit = None
        self.y_limit = None

    # draw a raceline
    def drawRaceline(self,img=None):
        raise NotImplementedError

    # draw a picture of the track
    def drawTrack(self, img=None,show=False):
        raise NotImplementedError

    def localTrajectory(self,state):
        raise NotImplementedError

    # NOTE universal function for all Track classes
    def setResolution(self,res):
        self.resolution = res
        return

    # determine if an coordinate is outside of track boundary, used in watchdog
    def isOutside(self,coord):
        grace = 1.0
        x,y = coord
        return x<-grace or y<-grace or x>self.x_limit+grace or y>self.y_limit+grace

    # NOTE plotting related
    def m2canvas(self,coord):
        x_new = int(np.clip(coord[0],0,self.x_limit) * self.resolution)
        y_new = int( (self.y_limit-np.clip(coord[1],0,self.y_limit)) * self.resolution )
        return (x_new,y_new)

    # draw a circle on canvas at coord
    def drawCircle(self, img, coord, radius_m, color = (0,0,0)):
        src = self.m2canvas(coord)
        radius_pix = int(radius_m * self.resolution)
        img = cv2.circle(img, src, radius_pix, color,-1)
        return img


    # draw a polynomial line defined in track space
    # points: a list of coordinates in format (x,y)
    def drawPolyline(self,points,img=None,lineColor=(0,0,255),thickness=3 ):

        if img is None:
            img = np.zeros([int(self.resolution*self.x_limit),int(self.resolution*self.y_limit),3],dtype='uint8')

        pts = [self.m2canvas(point) for point in points]
        for i in range(len(points)-1):
            p1 = np.array(pts[i])
            p2 = np.array(pts[i+1])
            if (pts[i] is None or pts[i+1] is None):
                continue
            img = cv2.line(img, tuple(p1),tuple(p2), color=lineColor ,thickness=thickness) 
        return img

    def drawTrajectory(self,traj_points,img=None,lineColor=(0,0,255),thickness=3 ):
        return self.drawPolyline(traj_points[:,1:3],img,lineColor,thickness)


    # draw ONE arrow, unit: meter, coord sys: dimensioned
    # source: source of arrow, in meter
    # orientation, radians from x axis, ccw positive
    # length: in pixels, though this is only qualitative
    def drawArrow(self,source, orientation, length, color=(0,0,0),thickness=2, img=None, show=False):
        if img is None:
            img = np.zeros([int(self.resolution*self.x_limit),int(self.resolution*self.y_limit),3],dtype='uint8')

        length = int(length)
        src = self.m2canvas(source)

        # y-axis positive direction in real world and cv plotting is reversed
        dest = (int(src[0] + cos(orientation)*length),int(src[1] - sin(orientation)*length))

        img = cv2.circle(img, src, 3, (0,0,0),-1)
        img = cv2.line(img, src, dest, color, thickness) 

        return img


    def prepareDiscretizedRaceline(self):
        '''
        depends on self.raceline_s, self.raceline_len_m
        '''
        ss = np.linspace(0,self.raceline_len_m,self.discretized_raceline_len)
        rr = splev(ss%self.raceline_len_m,self.raceline_s,der=0)
        drr = splev(ss%self.raceline_len_m,self.raceline_s,der=1)
        heading_vec = np.arctan2(drr[1],drr[0])
        vv = self.sToV(ss) 
        top_speed = 10
        vv[vv>top_speed] = top_speed

        # parameter, distance along track
        self.ss = ss
        self.raceline_points = np.array(rr)
        self.raceline_headings = heading_vec
        self.raceline_velocity = vv

        # describe track boundary as offset from raceline
        self.createBoundary()
        self.discretized_raceline = np.vstack([self.raceline_points,self.raceline_headings,vv, self.raceline_left_boundary, self.raceline_right_boundary]).T
        '''
        left = np.array(self.raceline_left_boundary)
        right = np.array(self.raceline_right_boundary)
        plt.plot(left+right)
        plt.show()
        breakpoint()
        '''
        return

    def createBoundary(self,show=False):
        '''
         construct a (self.discretized_raceline_len * 2) vector
         to record the left and right track boundary as an offset to the discretized raceline
         depends on self.preciseTrackBoundary(coord,heading)
        '''
        left_boundary = []
        right_boundary = []

        left_boundary_points = []
        right_boundary_points = []

        for i in range(self.discretized_raceline_len):
            # find normal direction
            coord = self.raceline_points[:,i]
            heading = self.raceline_headings[i]

            left, right = self.preciseTrackBoundary(coord,heading)
            left_boundary.append(left)
            right_boundary.append(right)

            # debug boundary points
            left_point = (coord[0] + left * cos(heading+np.pi/2),coord[1] + left * sin(heading+np.pi/2))
            right_point = (coord[0] + right * cos(heading-np.pi/2),coord[1] + right * sin(heading-np.pi/2))

            left_boundary_points.append(left_point)
            right_boundary_points.append(right_point)


            # DEBUG
            # plot left/right boundary
            '''
            left_point = (coord[0] + left * cos(heading+np.pi/2),coord[1] + left * sin(heading+np.pi/2))
            right_point = (coord[0] + right * cos(heading-np.pi/2),coord[1] + right * sin(heading-np.pi/2))
            img = self.drawTrack()
            img = self.drawRaceline(img = img)
            img = self.drawPoint(img,coord,color=(0,0,0))
            img = self.drawPoint(img,left_point,color=(0,0,0))
            img = self.drawPoint(img,right_point,color=(0,0,0))
            plt.imshow(img)
            plt.show()
            '''


        self.raceline_left_boundary = left_boundary
        self.raceline_right_boundary = right_boundary

        if (show):
            img = self.drawTrack()
            img = self.drawRaceline(img = img)
            img = self.drawPolyline(left_boundary_points,lineColor=(0,255,0),img=img)
            img = self.drawPolyline(right_boundary_points,lineColor=(0,0,255),img=img)
            plt.imshow(img)
            plt.show()
            return img
        return

    # save raceline to pickle file
    def save(self,filename=None):
        if filename is None:
            filename = "raceline.p"

        # check attribute types
        for attr in dir(self):
            if ( type(getattr(self,attr)) == types.FunctionType):
                self.print_warning(f' attribute .{attr} {getattr(self,attr)} may cause issue in saving if it is a lambda function')

        with open('./data/'+filename, 'wb') as f:
            pickle.dump(self,f)
        print_ok("track and raceline saved")

    # load quadratically smoothed raceline
    @staticmethod
    def load(filename=None):
        # get data folder abs path
        thisdir = os.path.dirname(os.path.abspath(__file__))
        basedir = os.path.dirname(thisdir)
        if filename is None:
            filename = "raceline.p"
        try:
            with open(basedir+'/data/'+filename, 'rb') as f:
                save = pickle.load(f)
        except FileNotFoundError:
            print_error("can't find "+filename+", run qpSmooth.py first")

        # restore save data
        print_ok('track loaded')
        return save
