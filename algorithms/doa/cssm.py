# Author: Eric Bezzam
# Date: July 15, 2016
from __future__ import division, print_function

from .music import *

class CSSM(MUSIC):
    """
    Class to apply the Coherent Signal-Subspace method (CSSM) [H. Wang and M. 
    Kaveh] for Direction of Arrival (DoA) estimation.

    .. note:: Run locate_source() to apply the CSSM algorithm.

    :param L: Microphone array positions. Each column should correspond to the 
    cartesian coordinates of a single microphone.
    :type L: numpy array
    :param fs: Sampling frequency.
    :type fs: float
    :param nfft: FFT length.
    :type nfft: int
    :param c: Speed of sound. Default: 343 m/s
    :type c: float
    :param num_src: Number of sources to detect. Default: 1
    :type num_src: int
    :param mode: 'far' or 'near' for far-field or near-field detection 
    respectively. Default: 'far'
    :type mode: str
    :param r: Candidate distances from the origin. Default: np.ones(1)
    :type r: numpy array
    :param azimuth: Candidate azimuth angles (in radians) with respect to x-axis.
    Default: np.linspace(-180.,180.,30)*np.pi/180
    :type azimuth: numpy array
    :param colatitude: Candidate colatitude angles (in radians) with respect to z-axis.
    Default is x-y plane search: np.pi/2*np.ones(1)
    :type colatitude: numpy array
    :param num_iter: Number of iterations for CSSM. Default: 5
    :type num_iter: int
    """
    def __init__(self, L, fs, nfft, c=343.0, num_src=1, mode='far', r=None,
        azimuth=None, colatitude=None, num_iter=5, **kwargs):

        MUSIC.__init__(self, L=L, fs=fs, nfft=nfft, c=c, num_src=num_src, 
            mode=mode, r=r, azimuth=azimuth, colatitude=colatitude, **kwargs)

        self.iter = num_iter
        self.init_test = np.zeros((self.M,self.grid.n_points),dtype=np.complex64)
        self.A0 = np.identity(self.M, dtype=np.complex64)
        self.Aj = np.identity(self.M, dtype=np.complex64)
        self.Tj = np.zeros((self.M,self.M), dtype=np.complex64)


    # @profile
    def _process(self, X):
        """
        Perform CSSM for given frame in order to estimate steered response 
        spectrum.
        """

        self.Pssl = np.zeros((self.num_freq,self.grid.n_points))

        # compute empirical cross correlation matrices
        C_hat = np.zeros([self.num_freq,self.M,self.M], dtype=np.complex64)
        for i, k in enumerate(self.freq_bins):
            C_hat[i,:,:] = np.dot(X[:,k,:],X[:,k,:].T.conj())/self.num_snap

        # compute initial estimates
        beta = []
        invalid = []

        # Find number of spatial spectrum peaks at each frequency band.
        # If there are less peaks than expected sources, leave the band out
        # Otherwise, store the location of the peaks.
        for i, k in enumerate(self.freq_bins):
            self.init_test[:] = np.dot(C_hat[i,:,:].conj().T, self.mode_vec[k,:,:])
            self.grid.set_values(np.sum(self.init_test*self.init_test.conj(), axis=0).real)
            idx = self.grid.find_peaks(k=self.num_src)

            if len(idx) < self.num_src:    # remove frequency
                invalid.append(i)
            else:
                beta.append(idx)


        # Here we remove the bands that had too few peaks
        self.freq_bins = np.delete(self.freq_bins, invalid)
        self.num_freq = self.num_freq - len(invalid)

        # compute reference frequency (take bin with max amplitude)
        f0 = np.argmax(np.sum(np.sum(abs(X[:,self.freq_bins,:]), axis=0),
            axis=1))
        f0 = self.freq_bins[f0]

        # iterate to find DOA, maximum number of iterations is 20
        i = 0
        while(i < self.iter):

            # coherent sum into self.CC
            self._coherent_sum(C_hat, f0, beta)

            # determine signal and noise subspace
            self.eigval[:],self.eigvec[:] = np.linalg.eig(self.CC)
            eigord = np.argsort(abs(self.eigval))
            self.noise_space[:] = self.eigvec[:,eigord[:-self.num_src]]

            # compute spatial spectrum
            self.music_test[:] = np.dot(self.noise_space.conj().T, 
                self.mode_vec[f0,:,:])
            self.grid.set_values(1./np.sum(np.multiply(self.music_test,self.music_test.conj()), axis=0).real)

            idx = self.grid.find_peaks(k=self.num_src)
            beta = np.tile(idx, (self.num_freq, 1))

            i += 1

    # @profile
    def _coherent_sum(self, C_hat, f0, beta):

        self.CC = np.zeros((self.M, self.M), dtype=np.complex64)

        # coherently sum frequencies
        for j, k in enumerate(self.freq_bins):

            lbj = len(beta[j])
            
            #
            self.A0[:,:lbj] = self.mode_vec[k,:,beta[j]].T
            self.Aj[:,:lbj] = self.mode_vec[f0,:,beta[j]].T

            self.Tj[:] = np.dot(self.A0, np.linalg.inv(self.Aj))

            self.CC[:] = self.CC + np.dot(np.dot(self.Tj,C_hat[j,:,:]),np.conjugate(self.Tj).T)
