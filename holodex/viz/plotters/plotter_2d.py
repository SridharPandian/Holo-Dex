import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from holodex.utils.network import create_push_socket, frequency_timer
from holodex.utils.files import *
from holodex.constants import *

def plot_line(X1, X2, Y1, Y2):
    plt.plot([X1, X2], [Y1, Y2])


class Plot2DMPHand(object):
    def __init__(self):
        # Thumb bound info
        self.thumb_bounds = None
        self.thumb_bounds_path = MP_THUMB_BOUNDS_PATH
        self.bound_update_counter = 0
        self._check_thumb_bounds()

        # Figure settings
        self.fig = plt.figure(figsize=(6, 6), dpi=60)

    def _check_thumb_bounds(self):
        if check_file(self.thumb_bounds_path):
            self.thumb_bounds = get_npz_data(self.thumb_bounds_path)[7:11]

    def _set_limits(self):
        plt.axis([-0.12, 0.12, -0.02, 0.2])

    def _draw_thumb_bounds(self):
        for idx in range(MP_THUMB_BOUND_VERTICES):
            plot_line(
                self.thumb_bounds[idx][0], 
                self.thumb_bounds[(idx + 1) % MP_THUMB_BOUND_VERTICES][0], 
                self.thumb_bounds[idx][1], 
                self.thumb_bounds[(idx + 1) % MP_THUMB_BOUND_VERTICES][1]
            )

    def draw_hand(self, X, Y):
        plt.plot(X, Y, 'ro')

        if self.thumb_bounds is not None:
            self._draw_thumb_bounds()

        # Drawing connections fromn the wrist
        for idx in MP_JOINTS['metacarpals']:
            plot_line(X[0], X[idx], Y[0], Y[idx])

        # Drawing knuckle to knuckle connections and knuckle to fingertip connections
        for key in ['knuckles', 'thumb', 'index', 'middle', 'ring', 'pinky']:
            for idx in range(len(MP_JOINTS[key]) - 1):
                plot_line(
                    X[MP_JOINTS[key][idx]], 
                    X[MP_JOINTS[key][idx + 1]], 
                    Y[MP_JOINTS[key][idx]], 
                    Y[MP_JOINTS[key][idx + 1]]
                )

    def draw(self, X, Y):
        # Setting the plot limits
        self._set_limits()

        # Resetting the thumb bounds
        if self.bound_update_counter % 10 == 0:
            self._check_thumb_bounds()
            self.bound_update_counter = 0
        else:
            self.bound_update_counter += 1

        # Plotting the lines to visualize the hand
        self.draw_hand(X, Y)
        plt.draw()

        # Resetting and Pausing the 3D plot
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        plt.cla()


class Plot2DOculusHand(object):
    def __init__(self, host, port):
        # Thumb bound info
        self.thumb_bounds = None
        self.thumb_bounds_path = VR_DISPLAY_THUMB_BOUNDS_PATH
        self.bound_update_counter = 0
        self._check_thumb_bounds()

        # Checking image storage path
        make_dir(os.path.join(CALIBRATION_FILES_PATH))

        # Figure settings
        self.fig = plt.figure(figsize=(6, 6), dpi=60)

        # Plot streamer settings
        self.socket = create_push_socket(host, port)
        self.frequency_timer = frequency_timer(60)


    def _check_thumb_bounds(self):
        if check_file(self.thumb_bounds_path):
            self.thumb_bounds = get_npz_data(self.thumb_bounds_path)

    def _set_limits(self):
        plt.axis([-0.12, 0.12, -0.02, 0.2])

    def _draw_thumb_bounds(self):
        for idx in range(VR_THUMB_BOUND_VERTICES):
            plot_line(
                self.thumb_bounds[idx][0], 
                self.thumb_bounds[(idx + 1) % VR_THUMB_BOUND_VERTICES][0], 
                self.thumb_bounds[idx][1], 
                self.thumb_bounds[(idx + 1) % VR_THUMB_BOUND_VERTICES][1]
            )
        
    def draw_hand(self, X, Y):
        plt.plot(X, Y, 'ro')

        if self.thumb_bounds is not None:
            self._draw_thumb_bounds()

        # Drawing connections fromn the wrist - 0
        for idx in OCULUS_JOINTS['metacarpals']:
            plot_line(X[0], X[idx], Y[0], Y[idx])

        # Drawing knuckle to knuckle connections and knuckle to finger connections
        for key in ['knuckles', 'thumb', 'index', 'middle', 'ring', 'pinky']:
            for idx in range(len(OCULUS_JOINTS[key]) - 1):
                plot_line(
                    X[OCULUS_JOINTS[key][idx]], 
                    X[OCULUS_JOINTS[key][idx + 1]], 
                    Y[OCULUS_JOINTS[key][idx]], 
                    Y[OCULUS_JOINTS[key][idx + 1]]
                )

    def draw(self, X, Y):
        # Setting the plot limits
        self._set_limits()

        # Resetting the thumb bounds
        if self.bound_update_counter % 10 == 0:
            self._check_thumb_bounds()
            self.bound_update_counter = 0
        else:
            self.bound_update_counter += 1

        # Plotting the lines to visualize the hand
        self.draw_hand(X, Y)
        plt.draw()

        # Saving and obtaining the plot
        plt.savefig(VR_2D_PLOT_SAVE_PATH)
        plot = cv2.imread(VR_2D_PLOT_SAVE_PATH)
        _, buffer = cv2.imencode('.jpg', plot, [int(cv2.IMWRITE_WEBP_QUALITY), 10])
        data = np.array(buffer).tobytes()
        self.socket.send(data)

        # Plotting
        plt.plot()

        # Resetting and pausing the 3D plot
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        plt.cla()