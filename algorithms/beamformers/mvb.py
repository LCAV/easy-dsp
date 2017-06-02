from __future__ import division, print_function

import numpy as np 
import matplotlib.pyplot as plt

import sys
from .utils import cart2polar, polar2cart
from transforms.dft import DFT

def H(A):
    """Returns the conjugate (Hermitian) transpose of a matrix."""
    return np.transpose(A).conj()

class MVB(object):

    def __init__(self, mic_pos, fs, nfft=256, c=343., direction=0., num_angles=60, num_snapshots=25, mu=0.):

        """
        :param noise_spec: noise spectrum, ambient + instrument
        :type noise_spec: numpy array
        :param mu: Factor (<1) for diagonal loading, i.e regularization
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
        self.nbin = self.nfft//2+1

        self.frequencies = np.arange(0, self.nbin)/float(self.nfft)*float(self.fs)
        self.mics_polar = cart2polar(self.L)

        # estimate of spatial cross correlation (cov), add overlapping?
        self.Rhat = np.zeros((self.nbin,self.M,self.M),dtype=np.complex64)
        self.mu = mu
        self.num_snapshots = num_snapshots
        self.frame_left = num_snapshots
        self.d = DFT(self.nfft,self.M, transform="fftw")
        self.ready = False   # done computing estimate of cov?
        self.frame = np.zeros((self.nfft,self.M),dtype='float32') # in case zero padding is needed

        # compute weights
        self.direction = direction
        self.weights = np.zeros((len(self.frequencies), self.M), 
            dtype=np.complex64)

    # @profile
    def estimate_cov(self, frame):

        self.frame[:frame.shape[0],:] = frame.astype(np.float32)
        self.d.analysis(self.frame)
        for k in range(self.nbin):
            h = np.array(self.d.X[k,:],ndmin=2)
            self.Rhat[k,:,:] += np.dot(np.conjugate(h.T),h)

        self.frame_left -= 1
        if self.frame_left==0:
            self.ready = True
            self.compute_weights()
            self.Rhat[k,:,:] /= self.num_snapshots
            self.compute_directivity()
    
    # @profile
    def compute_weights(self, direction=None):

        if direction is not None:
            self.direction = direction

        phi = self.direction*np.pi/180.
        src = polar2cart(np.array([1, phi]))

        # near-field
        # dist_m = np.linalg.norm(self.L - np.tile(src, (self.M,1)).T, 
        #     axis=0)
        # dist_c = np.linalg.norm(self.center-src)
        # dist_m = dist_m-dist_c

        # far-field
        dist_m = -np.dot(self.L.T, src)
        dist_m = dist_m - dist_m.min()

        for i, f in enumerate(self.frequencies):
            wavenum = 2*np.pi*f/self.c
            # mode_vecs = 1./dist_m * np.exp(-1j*wavenum*dist_m) # near field model
            mode_vecs = np.exp(-1j*wavenum*dist_m) # fair field model
            w = np.dot(H(mode_vecs), np.linalg.pinv(self.Rhat[i,:,:]+
                self.mu*np.diag(self.Rhat[i,:,:])*np.identity(self.M,dtype=complex)))
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





