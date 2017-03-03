# adapted from pyroomacoustics

from __future__ import division, print_function

import numpy as np 
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import utils, transforms

def H(A):
    """Returns the conjugate (Hermitian) transpose of a matrix."""
    return np.transpose(A).conj()



class DAS(object):

    def __init__(self, mic_pos, fs, nfft=256, c=343., direction=0.,
        num_angles=60):

        # only support even FFT length
        if nfft % 2 is 1:
            nfft += 1

        self.L = mic_pos[:2,:]    # only 2D beamforming!
        self.M = mic_pos.shape[1] # number of microphones
        self.center = np.mean(self.L, axis=1)
        self.nfft = int(nfft)
        self.fs = fs
        self.c = c

        self.frequencies = np.arange(0, self.nfft/2+1)/float(self.nfft)*float(self.fs)
        self.mics_polar = utils.cart2polar(self.L)

        # compute weights
        self.direction = direction
        self.weights = np.zeros((len(self.frequencies), self.M), 
            dtype=complex)
        self.compute_weights(direction)
        self.compute_directivity(num_angles)


    def compute_mode(self, freq, phi):

        X = utils.polar2cart(np.array([1, phi]))

        dist = np.linalg.norm(self.L - np.tile(X, (self.M,1)).T, axis=0)
        wavenum = 2 * np.pi * freq / self.c

        return np.exp(1j * wavenum * dist)

    def steering_vector_2D(self, frequency, phi):

        dist = 1.0
        phi = np.array([phi]).reshape(phi.size)

        # Assume phi and dist are measured from the array's center
        X = dist * np.array([np.cos(phi), np.sin(phi)]) + np.tile(self.center, (len(phi),1)).T

        D = np.dot(H(self.L), X)

        omega = 2 * np.pi * frequency

        return np.exp(1j * omega * D / self.c)


    def compute_weights(self, direction=None):

        if direction is not None:
            self.direction = direction

        phi = self.direction*np.pi/180.

        # for i, f in enumerate(self.frequencies):
        #     self.weights[:,i] = 1.0/self.M * self.compute_mode(f, phi)

        src = utils.polar2cart(np.array([1, phi]))
        dist_m = -np.dot(self.L.T, src)

        '''
        dist_m = np.linalg.norm(self.L - np.tile(src, (self.M,1)).T, 
            axis=0)
        dist_c = np.linalg.norm(self.center-src)
        dist_cent = dist_m-dist_c
        '''

        dist_cent = dist_m - dist_m.min()

        for i, f in enumerate(self.frequencies):
            wavenum = 2*np.pi*f/self.c
            self.weights[i,:] = np.exp(-1j * wavenum * dist_cent) / self.L.shape[1]



    def compute_directivity(self, num_angles=60):

        self.angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)

        resp = np.zeros((len(self.frequencies), num_angles), dtype=complex)

        for i, f in enumerate(self.frequencies):
            resp[i,:] = np.dot(H(self.weights[i,:]), 
                self.steering_vector_2D(f, self.angles))

        self.direct = np.abs(resp)**2
        self.direct /= self.direct.max()

        ## Tashev method unoptimized
        # self.angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
        # resp = np.zeros((len(self.frequencies), num_angles), dtype=complex)

        # for j, phi in enumerate(self.angles):
        #     direc = utils.polar2cart(np.array([1, phi]))
        #     dist_m = 1/np.linalg.norm(self.L - np.tile(direc, 
        #         (self.M,1)).T, axis=0)
        #     dist_c = np.linalg.norm(self.center-direc)
        #     dist_cent = dist_m-dist_c
        #     for i, f in enumerate(self.frequencies):
        #         wavenum = 2*np.pi*f/self.c
        #         D = dist_c * np.multiply(dist_m, 
        #             np.exp(-1j*wavenum*dist_cent))
        #         resp[i,j] = np.dot(H(self.weights[:,i]), D)

        # self.direct = np.abs(resp)**2
        # self.direct /= self.direct.max()


    def get_directivity(self, freq):

        freq_bin = int(np.round(float(freq)/self.fs*self.nfft))

        # print("Selected frequency: %.3f" % (float(freq_bin)/self.nfft*self.fs))

        return self.direct[freq_bin,:]


    def visualize_directivity(self, freq):

        freq_bin = int(np.round(float(freq)/self.fs*self.nfft))

        # print("Selected frequency: %.3f" % (float(freq_bin)/self.nfft*self.fs))

        ax = plt.subplot(111, projection='polar')
        angl = np.append(self.angles, 2*np.pi)
        resp = np.append(self.direct[freq_bin,:], self.direct[freq_bin,0])
        ax.plot(angl, resp)
        plt.show()

