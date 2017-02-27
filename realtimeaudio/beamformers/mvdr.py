from __future__ import division, print_function

import numpy as np 
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import utils

class MVDR(object):

    def __init__(self, mic_pos, fs, noise_spec, nfft=256, c=343., direction=0.,
        num_angles=60):

        """
        :param noise_spec: noise spectrum, ambient + instrument
        :type noise_spec: numpy array
        """

        # only support even FFT length
        if nfft % 2 is 1:
            nfft += 1

        # check size of noise spec!

        self.L = mic_pos[:2,:]    # only 2D beamforming!
        self.M = mic_pos.shape[1] # number of microphones
        self.center = np.mean(self.L, axis=1)
        self.nfft = int(nfft)
        self.fs = fs
        self.c = c
        self.noise_spec

        self.frequencies = np.arange(0, self.nfft/2+1)/float(self.nfft)*float(self.fs)
        self.mics_polar = utils.cart2polar(self.L)

        self.cross_noise = np.zeros((len(frequencies), self.M, self.M))
        self.build_noise_cross()

        # compute weights
        self.direction = direction
        self.weights = np.zeros((len(self.frequencies), self.M), 
            dtype=complex)
        self.compute_weights(direction)
        self.compute_directivity(num_angles)



    def build_noise_cross(self):

        # compute eucliden matrix
        dist = np.dot(mic_array.T, mic_array)
        md = np.tile(np.diag(dist), (self.M,1))
        dist = np.sqrt(md+md.T-2*dist)

        # compute cross spectrum of ambient noise
        for i, f in enumerate(self.frequencies):
            self.cross_noise[i] = self.noise_spec[i]*np.sinc(2*np.pi*dist/self.c)

    
    def compute_weights(self, direction=None):

        if direction is not None:
            self.direction = direction

        phi = self.direction*np.pi/180.
        src = utils.polar2cart(np.array([1, phi]))
        dist_m = np.linalg.norm(self.L - np.tile(src, (self.M,1)).T, 
            axis=0)

        mode_vecs = 



