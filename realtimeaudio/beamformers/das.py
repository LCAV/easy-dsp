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
        self.weights = np.zeros((self.M, len(self.frequencies)), 
            dtype=complex)
        self.compute_weights(direction)
        self.compute_directivity(num_angles)

        self.dft = transforms.DFT(nfft=self.nfft)


    def beamform(self, X):

        """
        :param X: DFT of input signals, where each column is from a different microphone
        :type X: numpy array
        """

        Y = np.diag(np.dot(X, self.weights))
        return self.dft.synthesis(Y)


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

        return np.exp(-1j * omega * D / self.c)


    def compute_weights(self, direction=None):

        if direction is None:
            direction = self.direction
        else:
            self.direction = direction

        phi = direction*np.pi/180

        for i, f in enumerate(self.frequencies):
            self.weights[:,i] = 1.0/self.M * self.compute_mode(f, phi)



    def compute_directivity(self, num_angles=60):

        self.angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)

        resp = np.zeros((len(self.frequencies), num_angles), dtype=complex)

        for i, f in enumerate(self.frequencies):
            resp[i,:] = np.dot(H(self.weights[:,i]), 
                self.steering_vector_2D(f, self.angles))

        self.direct = np.abs(resp)**2
        self.direct /= self.direct.max()


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













