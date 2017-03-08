# Author: Eric Bezzam
# Date: July 15, 2016

from doa import *

class MUSIC(DOA):
    """
    Class to apply MUltiple SIgnal Classication (MUSIC) direction-of-arrival 
    (DoA) for a particular microphone array.

    .. note:: Run locate_source() to apply the MUSIC algorithm.

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
    :param colatitude: Candidate elevation angles (in radians) with respect to z-axis.
    Default is x-y plane search: np.pi/2*np.ones(1)
    :type colatitude: numpy array
    """
    def __init__(self, L, fs, nfft, c=343.0, num_src=1, mode='far', r=None,
        azimuth=None, colatitude=None, **kwargs):

        DOA.__init__(self, L=L, fs=fs, nfft=nfft, c=c, 
            num_src=num_src, mode=mode, r=r, azimuth=azimuth, 
            colatitude=colatitude, **kwargs)

        self.Pssl = None
        self.CC = np.zeros((self.M, self.M), dtype=np.complex64)
        self.eigval = np.zeros(self.M, dtype=np.complex64)
        self.eigvec = np.zeros((self.M,self.M), dtype=np.complex64)
        self.noise_space = np.zeros((self.M,self.M-self.num_src),dtype=np.complex64)
        self.music_test = np.zeros((self.M-self.num_src,self.grid.n_points),dtype=np.complex64)
        
    # @profile
    def _process(self, X):
        """
        Perform MUSIC for given frame in order to estimate steered response 
        spectrum.
        """

        self.Pssl = np.zeros((self.num_freq,self.grid.n_points))

        # compute response for each frequency
        for i, k in enumerate(self.freq_bins):

            # estimate cross correlation
            self.CC[:] = np.dot(X[:,k,:], np.conj(X[:,k,:]).T) 

            # determine signal and noise subspace
            self.eigval[:],self.eigvec[:] = np.linalg.eig(self.CC)
            eigord = np.argsort(abs(self.eigval))
            self.noise_space[:] = self.eigvec[:,eigord[:-self.num_src]]

            # compute spatial spectrum
            self.music_test[:] = np.dot(self.noise_space.conj().T, 
                self.mode_vec[k,:,:])
            self.Pssl[i,:] = 1/np.sum(np.multiply(self.music_test,self.music_test.conj()), axis=0).real

        self.grid.set_values(np.sum(self.Pssl, axis=0)/self.num_freq)


    def _compute_spatial_spectrum(self, space, k):

        A = self.mode_vec[k,:,:]
        
        # using signal space
        # A_pow = np.sum(A*A.conj(), axis=0).real
        # music_test = np.dot(space.conj().T, A)
        # music_pow = np.sum(music_test*music_test.conj(), axis=0).real
        # music_pow -= A_pow

        # using noise space
        music_test = np.dot(space.conj().T, A)
        music_pow = np.sum(music_test*music_test.conj(), axis=0).real

        return 1/music_pow


    def _compute_correlation_matrices(self, X):
        C_hat = np.zeros([self.num_freq,self.M,self.M], dtype=complex)
        for i, k in enumerate(self.freq_bins):
            C_hat[i,:,:] = np.dot(X[:,k,:],X[:,k,:].T.conj())
        return C_hat/self.num_snap


    def _subspace_decomposition(self, R):

        # eigenvalue decomposition!
        w,v = np.linalg.eig(R)

        # sort out signal and noise subspace
        # Signal comprises the leading eigenvalues
        # Noise takes the rest
        eig_order = np.flipud(np.argsort(abs(w)))
        sig_space = eig_order[:self.num_src]
        noise_space = eig_order[self.num_src:]

        # eigenvalues
        ws = w[sig_space]
        wn = w[noise_space]

        # eigenvectors
        Es = v[:,sig_space]
        En = v[:,noise_space]

        return Es, En, ws, wn


