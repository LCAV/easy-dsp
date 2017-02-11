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

        DOA.__init__(self, L=L, fs=fs, nfft=nfft, c=c, num_src=num_src, 
            mode=mode, r=r, azimuth=azimuth, colatitude=colatitude, **kwargs)

        self.Pssl = None

    def _process(self, X):
        """
        Perform MUSIC for given frame in order to estimate steered response 
        spectrum.
        """

        self.Pssl = np.zeros((self.num_freq,self.grid.n_points))

        # estimate cross correlation
        # C_hat = self._compute_correlation_matrices(X)
        CC = []
        for k in self.freq_bins:
            X_k = X[:,k,:]
            CC.append( np.dot(X[:,k,:], np.conj(X[:,k,:]).T) )

        # compute response for each frequency
        for i in range(self.num_freq):
            k = self.freq_bins[i]

            # subspace decomposition
            # Es, En, ws, wn = self._subspace_decomposition(C_hat[i,:,:])
            # Es, En, ws, wn = self._subspace_decomposition(CC[i])

            w,v = np.linalg.eig(CC[i])
            eig_order = np.flipud(np.argsort(abs(w)))
            noise_space = eig_order[self.num_src:]
            En = v[:,noise_space]

            # compute spatial spectrum
            self.Pssl[i,:] = self._compute_spatial_spectrum(En,k)

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
        for i in range(self.num_freq):
            k = self.freq_bins[i]
            X_k = X[:,k,:]
            C_hat[i,:,:] = np.dot(X_k,X_k.T.conj())
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


    def plot_individual_spectrum(self):
        """
        Plot the steered response for each frequency.
        """

        # check if matplotlib imported
        if matplotlib_available is False:
            warnings.warn('Could not import matplotlib.')
            return

        # only for 2D
        if self.grid.dim == 3:
            pass
        else:
            warnings.warn('Only for 2D.')
            return

        # plot
        for k in range(self.num_freq):

            freq = float(self.freq_bins[k])/self.nfft*self.fs
            azimuth = self.grid.azimuth * 180 / np.pi

            plt.plot(azimuth, self.Pssl[k,0:len(azimuth)])

            plt.ylabel('Magnitude')
            plt.xlabel('Azimuth [degrees]')
            plt.xlim(min(azimuth),max(azimuth))
            plt.title('Steering Response Spectrum - ' + str(freq) + ' Hz')
            plt.grid(True)

