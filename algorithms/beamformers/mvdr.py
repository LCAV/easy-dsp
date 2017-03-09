from __future__ import division, print_function

import numpy as np 
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import utils

def H(A):
    """Returns the conjugate (Hermitian) transpose of a matrix."""
    return np.transpose(A).conj()

class MVDR(object):

    def __init__(self, mic_pos, fs, Rhat, nfft=256, c=343., direction=0., num_angles=60):

        """
        :param noise_spec: noise spectrum, ambient + instrument
        :type noise_spec: numpy array
        """

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

        # check size of noise spec!
        # assuming spatially homogeneous noise field, uncorrelated noise sources, omni mic --> p. 200 of Tashev
        # use spatial cov matrix, more in line with Frost
        self.Rhat = Rhat
        # self.cross_noise = np.zeros((len(self.frequencies), self.M, self.M), dtype=complex)
        # self.build_noise_cross()

        # compute weights
        self.direction = direction
        self.weights = np.zeros((len(self.frequencies), self.M), 
            dtype=complex)
        self.compute_weights(direction)
        self.compute_directivity(num_angles)



    def build_noise_cross(self):

        # compute eucliden matrix
        dist = np.dot(self.L.T, self.L)
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

        for i, f in enumerate(self.frequencies):
            wavenum = 2*np.pi*f/self.c
            mode_vecs = 1./dist_m * np.exp(-1j*wavenum*dist_m) # near field model
            w = np.dot(H(mode_vecs),
                np.linalg.pinv(self.Rhat[i,:,:]+0.1*np.identity(self.M,dtype=complex)))
            self.weights[i,:] = w / np.dot(w,mode_vecs)


    def steering_vector_2D(self, frequency, phi):

        dist = 1.0
        phi = np.array([phi]).reshape(phi.size)

        # Assume phi and dist are measured from the array's center
        X = dist * np.array([np.cos(phi), np.sin(phi)]) + np.tile(self.center, (len(phi),1)).T

        D = np.dot(H(self.L), X)

        omega = 2 * np.pi * frequency

        return np.exp(-1j * omega * D / self.c)


    def compute_directivity(self, num_angles=60):

        self.angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)

        resp = np.zeros((len(self.frequencies), num_angles), dtype=complex)

        for i, f in enumerate(self.frequencies):
            resp[i,:] = np.dot(H(self.weights[i,:]), 
                self.steering_vector_2D(f, self.angles))

        self.direct = np.abs(resp)**2
        self.direct /= self.direct.max()


    def visualize_directivity(self, freq):

        freq_bin = int(np.round(float(freq)/self.fs*self.nfft))

        # print("Selected frequency: %.3f" % (float(freq_bin)/self.nfft*self.fs))

        ax = plt.subplot(111, projection='polar')
        angl = np.append(self.angles, 2*np.pi)
        resp = np.append(self.direct[freq_bin,:], self.direct[freq_bin,0])
        ax.plot(angl, resp)
        plt.show()





