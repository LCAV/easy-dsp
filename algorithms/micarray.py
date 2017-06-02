# Author: Eric Bezzam
# Date: Feb 10, 2016

"""Class for microphone array."""

import numpy as np
import sys
from . import utils

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

class MicArray(object):
    """
    Methods
    --------
    visualize2D()
        Visualize microphone array on xy plane.
    """

    def __init__(self, X, fs=1.0):
        """
        Constructor for MicArray class.

        Parameters
        -----------
        L : numpy array
            Coordinates for each mic (each columns is a set of coordinates).
        """
        self.dim, self.M = X.shape
        self.X = X
        self.fs = fs

    def visualize2D(self, plt_show=False):
        """
        Visualize microphone array on xy plane.
        """
        if matplotlib_available == False:
            warnings.warn("Could not import matplotlib.")
            return
        # get coordinate
        x = self.X[0,:]
        y = self.X[1,:]
        diffX = (np.max(x)-np.min(x))*0.5
        diffY = (np.max(y)-np.min(y))*0.5
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y, marker='o', label='mics', c='r')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.grid(True)
        ax.set_title('2D Visualization of Microphone Array')
        # plt.scatter(x, y, marker='o', label='mics', c='r')
        # plt.axis([np.min(x)-diffX, np.max(x)+diffX, np.min(y)-diffY, 
        #     np.max(y)+diffY])
        # plt.grid(True)
        # plt.xlabel('x [m]')
        # plt.ylabel('y [m]')
        # plt.title('2D Visualization of Microphone Array')
        if plt_show: plt.show()
        return fig

    def visualize3D(self, plt_show=False):
        """
        Visualize microphone array in 3 dimensions.
        """
        if matplotlib_available == False:
            warnings.warn("Could not import matplotlib.")
            return
        # get coordinate
        x = self.X[0,:]
        y = self.X[1,:]
        z = self.X[2,:]
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        plt.grid(True)
        ax.set_title('3D Visualization of Microphone Array')
        if plt_show: plt.show()
