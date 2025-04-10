# class to display, manipulate, and save pgm maps
'''
usage: open a pgm file, use left, right, middle mouse button to clean up the map so that the track
surface is a closed region
Then, click the "Label Track Surface" button to enter track surface mode, and click anywhere
inside the track surface to paint it, self.track_surface_pixels will contain the corresponding pixels
'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button
from enum import Enum

from common import *

class Mode(Enum):
    NONE = 0
    TRACK = 1

class Pgm(PrintObject):
    def __init__(self):
        # w,h
        self.size = None
        self.depth = None
        # W * H 2D array of type uint8, 0 = occupied, 254/255 = free, other value = unknown
        self.data = None
        self.magic = None

        self.last_click = (-1,-1)
        self.mode = Mode.NONE
        self.track_surface_pixels = None


    def readLine(self, file):
        ''' read the next line that's not a comment (start with #) '''
        while True:
            line = file.readline().decode().strip()
            if (line[0] != '#'):
                return line

    def readPgm(self, filename):
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
            return (magic, width, height, depth, self.data)

    def show(self):
        ''' show the pgm '''
        plt.imshow(self.data)
        plt.show()
        return

    def select(self):
        ''' allow user to interactively select pixles for labeling'''
        fig, ax = plt.subplots()
        img_display = ax.imshow(self.data)

        def onclick(event):
            if fig.canvas.manager.toolbar.mode != '':
                return
            if not event.inaxes == ax:
                return
            if (self.mode == Mode.NONE):
                modifyCallback(event)
            elif (self.mode == Mode.TRACK):
                trackCallback(event)

        def onmotion(event):
            return
        def onrelease(event):
            return

        def modifyCallback(event):
            x = int(event.xdata+0.5)
            y = int(event.ydata+0.5)
            print(f'user clicked {x,y}')
            if (event.button == MouseButton.LEFT):
                self.data[y,x] = 0
            if (event.button == MouseButton.RIGHT):
                self.data[y,x] = 205
            if (event.button == MouseButton.MIDDLE):
                self.data[y,x] = 254
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
                    if (self.data[neibor[1], neibor[0]] != 0 and not neibor in pixels and not neibor in frontier):
                        # if not an obstacle
                        frontier.add(neibor)

            cols, rows = zip(*pixels)
            self.data[rows, cols] = 100
            img_display.set_data(self.data)
            plt.draw()
            self.track_surface_pixels = pixels
            return

        def toggleTrackModeCallback(event):
            if (self.mode == Mode.NONE):
                self.mode = Mode.TRACK
                self.print_info('Track surface labeling mode')
            else:
                self.mode = Mode.NONE
                self.print_info('Map Repair mode')


        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        cid = fig.canvas.mpl_connect('motion_notify_event', onmotion)
        cid = fig.canvas.mpl_connect('button_release_event', onrelease)

        ax_button = plt.axes([0,0,0.2,0.05])
        track_button = Button(ax_button, 'Label Track Surface')
        track_button.on_clicked(toggleTrackModeCallback)
        plt.show()

        return



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
    main = Pgm()
    main.readPgm('map_house.pgm')
    main.select()


