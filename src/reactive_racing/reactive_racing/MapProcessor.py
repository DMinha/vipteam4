# class to display, manipulate, and save pgm maps
'''
usage: open a pgm file, use left, right, middle mouse button to clean up the map so that the track
surface is a closed region
Then, click the "Label Track Surface" button to enter track surface mode, and click anywhere
inside the track surface to paint it, self.track_surface_pixels will contain the corresponding pixels
'''


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button
from enum import Enum
from PIL import Image
import yaml

from .common import *

class Mode(Enum):
    ''' modes for the interactive map display '''

    REPAIR = 0 # repair map, manually label pixels as free/occupied/unknown
    TRACK = 1 # label track surface
    REF = 2 # label a reference path

class MapProcessor(PrintObject):
    def __init__(self):
        # w,h
        self.size = None
        self.depth = None
        # W * H 2D array of type uint8, 0 = occupied, 254/255 = free, other value = unknown
        self.data = None
        self.magic = None
        self.img_filename = None
        self.yaml_filename = None

        self.last_click = (-1,-1)
        self.mode = Mode.REPAIR
        self.track_surface_pixels = None
        self.reference_path_pixels = None


    def readLine(self, file):
        ''' read the next line that's not a comment (start with #) '''
        while True:
            line = file.readline().decode().strip()
            if (line[0] != '#'):
                return line

    def readYaml(self, filename):
        ext = os.path.splitext(filename)[1]
        if (ext != '.yaml'):
            self.print_error(f' file extension is {ext} but yaml processor is called ')
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)
        self.print_info(data)
        if (data['image'] != self.img_filename):
            self.print_error(f' yaml-image file is inconsistent with loaded img file, or readYaml() is called before readPgm/readPng')
        self.resolution = data['resolution']
        self.origin = data['origin']
        self.occupied_thresh = 254 * data['occupied_thresh']
        self.free_thresh = 254 * data['free_thresh']
        return


    def readPgm(self, filename):
        ext = os.path.splitext(filename)[1]
        if (ext != '.pgm'):
            self.print_error(f' file extension is {ext} but pgm processor is called ')
        with open(filename, 'rb') as pgmfile:
            magic = self.readLine(pgmfile)
            if magic not in ('P2', 'P5'):
                raise ValueError("Not a PGM file: " + filename)
            size = self.readLine(pgmfile).split()
            width, height = int(size[0]), int(size[1])
            depth = int(self.readLine(pgmfile))
            if magic == 'P2':
                # ASCII
                data = []
                for y in range(height):
                    row = list(map(int, pgmfile.readline().decode().strip().split()))
                    data.extend(row)
            elif magic == 'P5':
                # raw
                data = bytearray(pgmfile.read())

            self.size = (width, height)
            self.data = np.array(data, dtype=np.uint8).reshape((height, width))
            self.depth = depth
            self.magic = magic
            self.img_filename = os.path.basename(filename)
            return

    def readPng(self, filename):
        ext = os.path.splitext(filename)[1]
        if (ext != '.png' and ext != '.PNG'):
            self.print_error(f' file extension is {ext} but png processor is called ')
        img = Image.open(filename)
        self.data = np.array(img, dtype=np.uint8)
        self.size = self.data.shape
        self.img_filename = os.path.basename(filename)
        return


    def show(self):
        ''' show the pgm '''
        plt.imshow(self.data, cmap='gray')
        plt.show()
        return

    def interactiveMapEdit(self):
        ''' allow user to interactively select pixles for labeling'''
        fig, ax = plt.subplots()
        path, = ax.plot([],[],'-')
        img_display = ax.imshow(self.data, cmap='gray')
        title = ax.set_title('Map Repair Mode')

        def onclick(event):
            if fig.canvas.manager.toolbar.mode != '':
                return
            if not event.inaxes == ax:
                return
            if (self.mode == Mode.REPAIR):
                modifyCallback(event)
            elif (self.mode == Mode.TRACK):
                trackCallback(event)
            elif (self.mode == Mode.REF):
                refPathCallback(event)

        def onmotion(event):
            if fig.canvas.manager.toolbar.mode != '':
                return
            if not event.inaxes == ax or event.button == None:
                return
            if (self.mode == Mode.REPAIR):
                modifyCallback(event)
            elif (self.mode == Mode.REF):
                refPathCallback(event)
            return

        def onrelease(event):
            return

        def modifyCallback(event):
            x = int(event.xdata+0.5)
            y = int(event.ydata+0.5)
            #print(f'user clicked {x,y}')
            neibor = lambda x,y : [(x+0,y+0), (x-1,y), (x+1,y), (x-1,y-1), (x,y+1), (x,y-1), (x+1,y+1), (x+1,y-1), (x-1,y+1)]
            fill_val = None
            if (event.button == MouseButton.LEFT): # obstacle
                fill_val = 0
            if (event.button == MouseButton.RIGHT): # unknown
                fill_val = (self.free_thresh + self.occupied_thresh) // 2
            if (event.button == MouseButton.MIDDLE): # clear
                fill_val = 254
            if fill_val is None:
                return
            for pnt in neibor( y,x ):
                self.data[pnt[0], pnt[1]] = fill_val
            img_display.set_data(self.data)
            plt.draw()


        def trackCallback(event):
            ''' paint all connected pixels as drivable surface'''
            x = int(event.xdata+0.5)
            y = int(event.ydata+0.5)
            pixels = set()
            frontier = {(x,y)}

            while (len(frontier) > 0):
                x,y = frontier.pop()
                pixels.add((x,y))
                neibors = [ (x+1, y), (x-1, y), (x,y+1), (x,y-1) ]
                for neibor in neibors:
                    if (self.data[neibor[1], neibor[0]] > self.free_thresh and not neibor in pixels and not neibor in frontier):
                        # if not an obstacle
                        frontier.add(neibor)

            cols, rows = zip(*pixels)
            # NOTE don't let this mess up the saved map
            self.data[rows, cols] = 170
            img_display.set_data(self.data)
            self.mode = Mode.REPAIR
            self.print_info('track complete, back to Map Repair mode')
            title = ax.set_title('Map Repair Mode')
            self.track_surface_pixels = pixels
            plt.draw()

            return

        def refPathCallback(event):
            x = int(event.xdata+0.5)
            y = int(event.ydata+0.5)
            if (event.button == MouseButton.LEFT): # obstacle
                self.data[y,x] = 50
                self.reference_path_pixels.append( (x,y) )
                img_display.set_data(self.data)
                ref_path = np.array(self.reference_path_pixels)
                path.set_data(ref_path[:,0], ref_path[:,1])
                plt.draw()
                return

        def toggleTrackModeCallback(event):
            if (self.mode == Mode.REPAIR):
                self.mode = Mode.TRACK
                self.print_info('Track surface labeling mode')
                title = ax.set_title('Track surface labeling mode')
            else:
                self.mode = Mode.REPAIR
                self.print_info('Map Repair mode')
                title = ax.set_title('Map Repair Mode')
            fig.canvas.draw_idle()

        def toggleRefPathModeCallback(event):
            if (self.mode == Mode.REPAIR):
                self.mode = Mode.REF
                #  clear ref path
                self.reference_path_pixels = []
                self.print_info('Reference Path Mode')
                title = ax.set_title('Reference Path Mode')
            else:
                self.mode = Mode.REPAIR
                self.print_info('Map Repair mode')
                title = ax.set_title('Map Repair Mode')
            fig.canvas.draw_idle()


        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        cid = fig.canvas.mpl_connect('motion_notify_event', onmotion)
        cid = fig.canvas.mpl_connect('button_release_event', onrelease)

        ax_button = plt.axes([0,0,0.2,0.05])
        track_button = Button(ax_button, 'Label Track Surface')
        track_button.on_clicked(toggleTrackModeCallback)

        ax_button = plt.axes([0.2,0,0.2,0.05])
        ref_path_button = Button(ax_button, 'Label Reference Path')
        ref_path_button.on_clicked(toggleRefPathModeCallback)

        plt.show()
        return

    def pixel2m(self, coord):
        ''' pixel coordinate to world coordinate '''
        return (coord[0] * self.resolution + self.origin[0], coord[1] * self.resolution + self.origin[1])

    def isValidTrack(self, coord):
        ''' check if a coordinate : (x,y) world frame, unit:m, is in the drivable track surface as defined with map'''
        # TODO I'm not sure what "origin" means exactly
        # sef.data.shape = (height, width)
        # during display in plt, upper left is (0,0)
        # We will ignore orientation, and just consider
        pixel_coord = ( (coord[0] - self.origin[0])//self.resolution, (coord[1] - self.origin[1])//self.resolution)
        return pixel_coord in self.track_surface_pixels

    def displayPath(self, coord_vecs):
        ''' display path on map, coord_vec is a list of (x,y) in world coordinate unit:m '''
        fig, ax = plt.subplots()
        img_display = ax.imshow(self.data, cmap='gray')

        for coord_vec in coord_vecs:
            pixel_coord_vec = []
            for coord in coord_vec:
                pixel_coord = ( (coord[0] - self.origin[0])/self.resolution, (coord[1] - self.origin[1])/self.resolution)
                pixel_coord_vec.append(pixel_coord)
            pixel_coord_vec = np.array(pixel_coord_vec)
            ax.plot(pixel_coord_vec[:,0], pixel_coord_vec[:,1],'-')

        plt.show()



    '''
    def save_pgm(filename, data, magic='P2', depth=255):
       with open(filename, 'wb') as pgmfile:
            pgmfile.write(f'{magic}\n'.encode())
            pgmfile.write(f'{data.shape[1]} {data.shape[0]}\n'.encode())
            pgmfile.write(f'{depth}\n'.encode())
            if magic == 'P2':
                np.savetxt(pgmfile, data, fmt='%d')
            elif magic == 'P5':
                pgmfile.write(data.tobytes())
    '''

if __name__ ==  '__main__':
    main = MapProcessor()
    main.readPng('maps/f1tenth_maps/maps/Oschersleben_map.png')
    main.readYaml('maps/f1tenth_maps/maps/Oschersleben_map.yaml')
    #main.readPgm('maps/my_map.pgm')
    #main.readYaml('maps/my_map.yaml')
    main.interactiveMapEdit()
    print(main.reference_path_pixels)


