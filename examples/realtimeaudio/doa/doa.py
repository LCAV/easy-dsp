# Author: Eric Bezzam
# Date: Feb 15, 2016
from __future__ import division, print_function

"""Direction of Arrival (DoA) estimation."""

import numpy as np
import math, sys
import warnings
from abc import ABCMeta, abstractmethod

try:
    import matplotlib as mpl

    matplotlib_available = True
except ImportError:
    matplotlib_available = False

if matplotlib_available:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

from grid import GridCircle, GridSphere

tol = 1e-14


class ModeVector(object):
    '''
    This class fakes a very large look-up table of mode vectors
    '''

    def __init__(self, L, fs, nfft, c, grid, mode='far', precompute=False):

        if (nfft % 2 == 1):
            raise ValueError('Signal length must be even.')

        # this flag controls if the look-up table should be stored
        # or computed on the fly
        self.precompute = precompute

        # short hands for propagation vectors, upped to 3D array
        p_x = grid.x[None, None, :]
        p_y = grid.y[None, None, :]
        p_z = grid.z[None, None, :]

        # short hands for microphone locations, upped to 3D array
        r_x = L[0, None, :, None]
        r_y = L[1, None, :, None]

        if L.shape[0] == 3:
            r_z = L[2, None, :, None]
        else:
            r_z = np.zeros((1, L.shape[1], 1))

        # Here we compute the time of flights from source candidate locations
        # to microphones
        if mode == 'near':
            # distance
            dist = np.sqrt((p_x - r_x) ** 2 + (p_y - r_y) ** 2 + (p_z - r_z) ** 2)

        elif mode == 'far':
            # projection
            dist = (p_x * r_x) + (p_y * r_y) + (p_z * r_z)

        # shape (nfft // 2 + 1)
        self.tau = dist / c

        # shape (1, num_mics, grid_size)
        self.omega = 2 * np.pi * fs * np.arange(nfft // 2 + 1) / nfft

        if precompute:
            self.mode_vec = np.exp(1j * self.omega[:, None, None] * self.tau)
        else:
            self.mode_vec = None

    def __getitem__(self, ref):

        # If the look up table was precomputed
        if self.precompute:
            return self.mode_vec[ref]

        # Otherwise compute values on the fly
        if isinstance(ref[1], int) and isinstance(ref[2], int):
            w = self.omega[ref[0]]
        elif isinstance(ref[1], int) or isinstance(ref[2], int):
            w = self.omega[ref[0], None]
        else:
            w = self.omega[ref[0], None, None]

        if isinstance(ref[0], int):
            tref0 = 0
        else:
            tref0 = slice(None, None, None)

        if len(ref) == 1:
            return np.exp(1j * w * self.tau[tref0, :, :])
        elif len(ref) == 2:
            return np.exp(1j * w * self.tau[tref0, ref[1], :])
        elif len(ref) == 3:
            return np.exp(1j * w * self.tau[tref0, ref[1], ref[2]])
        else:
            raise ValueError('Too many axis')


class DOA(object):
    """

    Abstract parent class for Direction of Arrival (DoA) algorithms. After 
    creating an object (SRP, MUSIC, CSSM, WAVES, or TOPS), run locate_source to
    apply the corresponding algorithm.

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
    :param n_grid: If azimuth and colatitude are not specified, we will create a grid with
    so many points. Default is 360.
    :type n_grid: int
    :param dim: The dimension of the problem. Set dim=2 to find sources on the circle (x-y plane).
    Set dim=3 to search on the whole sphere.
    :type dim: int
    """

    __metaclass__ = ABCMeta

    def __init__(self, L, fs, nfft, c=343.0, num_src=1, mode='far', r=None,
                 azimuth=None, colatitude=None, n_grid=None, dim=2, *args, **kwargs):

        if dim > L.shape[0]:
            raise ValueError('Microphones locations missing dimensions.')

        self.L = L  # locations of mics
        self.fs = fs  # sampling frequency
        self.c = c  # speed of sound
        self.M = L.shape[1]  # number of microphones
        self.D = L.shape[0]  # number of dimensions (x,y,z)
        self.num_snap = None  # number of snapshots

        self.nfft = nfft
        self.max_bin = int(self.nfft // 2) + 1
        self.freq_bins = None
        self.freq_hz = None
        self.num_freq = None

        self.num_src = self._check_num_src(num_src)
        self.sources = np.zeros([self.D, self.num_src])
        self.src_idx = np.zeros(self.num_src, dtype=np.int)
        self.azimuth_recon = None
        self.colatitude_recon = None
        self.alpha_recon = None

        self.mode = mode
        if self.mode is 'far':
            self.r = np.ones(1)
        elif r is None:
            self.r = np.ones(1)
            self.mode = 'far'
        else:
            self.r = r
            if r == np.ones(1):
                mode = 'far'

        # Set the dimension of the problem
        if dim != 2 and dim != 3:
            raise ValueError("The dimension must be 2 or 3.")
        else:
            self.dim = dim

        # Some logic to create a grid for search
        self.grid = None
        if azimuth is None and colatitude is None:

            # Use a default grid size
            if dim == 2:

                if n_grid is None:
                    n_grid = 360

                self.grid = GridCircle(n_points=n_grid)

            elif dim == 3:

                if n_grid is None:
                    n_grid = 180 * 90

                self.grid = GridSphere(n_points=n_grid)

        elif azimuth is None and colatitude is not None:

            raise ValueError("Azimuth should always be specified.")

        else:

            if dim == 2:

                if colatitude is not None:
                    warnings.warn('Colatitude is ignored for 2D problems.')

                self.grid = GridCircle(azimuth=azimuth)

            elif dim == 3:

                if azimuth.ndim != 1:
                    raise ValueError('Azimuth should be a 1D ndarray.')

                if colatitude is None:
                    warnings.warn('Colatitude is not specified. Setting all colatitude to pi / 2.')

                    colatitude = (np.pi / 2) * np.ones(azimuth.shape[0])
                    grid_points = np.vstack((azimuth, colatitude))

                    self.grid = GridSphere(spherical_points=grid_points)

                else:

                    # when both azimuth and theta are specified,
                    # we assume we want the cartesian product
                    A, C = np.meshgrid(np.unique(azimuth), np.unique(colatitude))
                    grid_points = np.vstack((A.flatten(), C.flatten()))

                    self.grid = GridSphere(spherical_points=grid_points)

        # spatial spectrum / dirty image (FRI)
        self.P = None

        # build lookup table to candidate locations from r, azimuth, colatitude 

        # from .fri import FRI

        # if not isinstance(self, FRI):
        self.mode_vec = ModeVector(self.L, self.fs, self.nfft, self.c, self.grid)

    def locate_sources(self, X, num_src=None, freq_range=[500.0, 4000.0],
                       freq_bins=None, freq_hz=None):
        """
        Locate source(s) using corresponding algorithm.

        :param X: Set of signals in the frequency (RFFT) domain for current 
        frame. Size should be M x F x S, where M should correspond to the 
        number of microphones, F to nfft/2+1, and S to the number of snapshots 
        (user-defined). It is recommended to have S >> M.
        :type X: numpy array
        :param num_src: Number of sources to detect. Default is value given to 
        object constructor.
        :type num_src: int
        :param freq_range: Frequency range on which to run DoA: [fmin, fmax].
        :type freq_range: list of floats, length 2
        :param freq_bins: List of individual frequency bins on which to run 
        DoA. 
        If defined by user, it will **not** take into consideration freq_range 
        or freq_hz.
        :type freq_bins: list of int
        :param freq_hz: List of individual frequencies on which to run DoA. If 
        defined by user, it will **not** take into consideration freq_range.
        :type freq_hz: list of floats
        """

        # check validity of inputs
        if num_src is not None and num_src != self.num_src:
            self.num_src = self._check_num_src(num_src)
            self.sources = np.zeros([self.num_src, self.D])
            self.src_idx = np.zeros(self.num_src, dtype=np.int)
            self.angle_of_arrival = None
        if X.shape[0] != self.M:
            raise ValueError('Number of signals (rows) does not match the \
                number of microphones.')
        if X.shape[1] != self.max_bin:
            raise ValueError("Mismatch in FFT length.")
        self.num_snap = X.shape[2]

        # frequency bins on which to apply DOA
        if freq_bins is not None:
            self.freq_bins = freq_bins
        elif freq_hz is not None:
            self.freq_bins = [int(np.round(f / self.fs * self.nfft))
                              for f in freq_bins]
        else:
            freq_range = [int(np.round(f / self.fs * self.nfft))
                          for f in freq_range]
            self.freq_bins = np.arange(freq_range[0], freq_range[1])

        self.freq_bins = self.freq_bins[self.freq_bins < self.max_bin]
        self.freq_bins = self.freq_bins[self.freq_bins >= 0]
        self.freq_hz = self.freq_bins * float(self.fs) / float(self.nfft)
        self.num_freq = len(self.freq_bins)

        # search for DoA according to desired algorithm

        # initialize the grid value to zero
        self.grid.set_values(0.)

        # Run the algorithm
        self._process(X)

        # locate sources
        # from .fri import FRI
        # if not isinstance(self, FRI):

        self.src_idx = self.grid.find_peaks(k=self.num_src)

        self.num_src = len(self.src_idx)

        if self.dim == 2:
            self.azimuth_recon = self.grid.azimuth[self.src_idx]
        elif self.dim == 3:
            self.azimuth_recon = self.grid.azimuth[self.src_idx]
            self.colatitude_recon = self.grid.colatitude[self.src_idx]

        #self.grid.values /= np.max(self.grid.values)

    def polar_plt_dirac(self, phi_ref=None, alpha_ref=None, save_fig=False,
                        file_name=None, plt_dirty_img=True):
        """
        Generate polar plot of DoA results.

        :param phi_ref: True direction of sources (in radians).
        :type phi_ref: numpy array
        :param alpha_ref: Estimated amplitude of sources.
        :type alpha_ref: numpy array
        :param save_fig: Whether or not to save figure as pdf.
        :type save_fig: bool
        :param file_name: Name of file (if saved). Default is 
        'polar_recon_dirac.pdf'
        :type file_name: str
        :param plt_dirty_img: Whether or not to plot spatial spectrum or 
        'dirty image' in the case of FRI.
        :type plt_dirty_img: bool
        """

        if self.dim != 2:
            raise ValueError('This function only handles 2D problems.')

        azimuth_recon = self.azimuth_recon
        num_mic = self.M
        phi_plt = self.grid.azimuth

        # determine amplitudes
        # from .fri import FRI
        # if not isinstance(self, FRI):  # use spatial spectrum

        dirty_img = self.grid.values
        alpha_recon = self.grid.values[self.src_idx]
        alpha_ref = alpha_recon

        # else:  # create dirty image

        #     dirty_img = self._gen_dirty_img()
        #     alpha_recon = np.mean(np.abs(self.alpha_recon), axis=1)
        #     alpha_recon /= alpha_recon.max()
        #     if alpha_ref is None:  # non-simulated case
        #         alpha_ref = alpha_recon

        # plot
        fig = plt.figure(figsize=(5, 4), dpi=90)
        ax = fig.add_subplot(111, projection='polar')
        base = 1.
        height = 10.
        blue = [0, 0.447, 0.741]
        red = [0.850, 0.325, 0.098]

        if phi_ref is not None:
            if alpha_ref.shape[0] < phi_ref.shape[0]:
                alpha_ref = np.concatenate((alpha_ref, np.zeros(phi_ref.shape[0] -
                                                                alpha_ref.shape[0])))

            # match detected with truth
            recon_err, sort_idx = polar_distance(azimuth_recon, phi_ref)
            if self.num_src > 1:
                azimuth_recon = azimuth_recon[sort_idx[:, 0]]
                alpha_recon = alpha_recon[sort_idx[:, 0]]
                phi_ref = phi_ref[sort_idx[:, 1]]
                alpha_ref = alpha_ref[sort_idx[:, 1]]
            elif phi_ref.shape[0] > 1:  # one detected source
                alpha_ref[sort_idx[1]] = alpha_recon

            # markers for original doa
            K = len(phi_ref)
            ax.scatter(phi_ref, base + height * alpha_ref, c=np.tile(blue,
                                                                     (K, 1)), s=70, alpha=0.75, marker='^',
                       linewidths=0,
                       label='original')

            # stem for original doa
            if K > 1:
                for k in range(K):
                    ax.plot([phi_ref[k], phi_ref[k]], [base, base +
                                                       height * alpha_ref[k]], linewidth=1.5, linestyle='-',
                            color=blue, alpha=0.6)
            else:
                ax.plot([phi_ref, phi_ref], [base, base + height * alpha_ref],
                        linewidth=1.5, linestyle='-', color=blue, alpha=0.6)

        K_est = azimuth_recon.size

        # markers for reconstructed doa
        ax.scatter(azimuth_recon, base + height * alpha_recon, c=np.tile(red,
                                                                         (K_est, 1)), s=100, alpha=0.75, marker='*',
                   linewidths=0,
                   label='reconstruction')

        # stem for reconstructed doa
        if K_est > 1:
            for k in range(K_est):
                ax.plot([azimuth_recon[k], azimuth_recon[k]], [base, base +
                                                               height * alpha_recon[k]], linewidth=1.5, linestyle='-',
                        color=red, alpha=0.6)

        else:
            ax.plot([azimuth_recon, azimuth_recon], [base, base + height * alpha_recon],
                    linewidth=1.5, linestyle='-', color=red, alpha=0.6)

            # plot the 'dirty' image
        if plt_dirty_img:
            dirty_img = np.abs(dirty_img)
            min_val = dirty_img.min()
            max_val = dirty_img.max()
            dirty_img = (dirty_img - min_val) / (max_val - min_val)

            # we need to make a complete loop, copy first value to last
            c_phi_plt = np.r_[phi_plt, phi_plt[0]]
            c_dirty_img = np.r_[dirty_img, dirty_img[0]]
            ax.plot(c_phi_plt, base + height * c_dirty_img, linewidth=1,
                    alpha=0.55, linestyle='-', color=[0.466, 0.674, 0.188],
                    label='spatial spectrum')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:3], framealpha=0.5,
                  scatterpoints=1, loc=8, fontsize=9,
                  ncol=1, bbox_to_anchor=(0.9, -0.17),
                  handletextpad=.2, columnspacing=1.7, labelspacing=0.1)

        # ax.set_xlabel(r'azimuth $\bm{\varphi}$', fontsize=11)
        ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
        ax.xaxis.set_label_coords(0.5, -0.11)
        ax.set_yticks(np.linspace(0, 1, 2))
        ax.xaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle=':')
        ax.yaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle='--')
        ax.set_ylim([0, 1.05 * (base + height)])
        if save_fig:
            if file_name is None:
                file_name = 'polar_recon_dirac.pdf'
            plt.savefig(file_name, format='pdf', dpi=300, transparent=True)

    def _check_num_src(self, num_src):
        # # check validity of inputs
        # if num_src > self.M:
        #     warnings.warn('Number of sources cannot be more than number of \
        #         microphones. Changing number of sources to ' +
        #         str(self.M) + '.')
        #     num_src = self.M
        if num_src < 1:
            warnings.warn('Number of sources must be at least 1. Changing \
                number of sources to 1.')
            num_src = 1
        valid = num_src
        return valid


# ------------------Miscellaneous Functions---------------------#

def spher2cart(r, azimuth, colatitude):
    """
    Convert a spherical point to cartesian coordinates.
    """
    # convert to cartesian
    x = r * np.cos(azimuth) * np.sin(colatitude)
    y = r * np.sin(azimuth) * np.sin(colatitude)
    z = r * np.cos(colatitude)
    return np.array([x, y, z])


def polar_distance(x1, x2):
    """
    Given two arrays of numbers x1 and x2, pairs the cells that are the
    closest and provides the pairing matrix index: x1(index(1,:)) should be as
    close as possible to x2(index(2,:)). The function outputs the average of 
    the absolute value of the differences abs(x1(index(1,:))-x2(index(2,:))).
    :param x1: vector 1
    :param x2: vector 2
    :return: d: minimum distance between d
             index: the permutation matrix
    """
    x1 = np.reshape(x1, (1, -1), order='F')
    x2 = np.reshape(x2, (1, -1), order='F')
    N1 = x1.size
    N2 = x2.size
    diffmat = np.arccos(np.cos(x1 - np.reshape(x2, (-1, 1), order='F')))
    min_N1_N2 = np.min([N1, N2])
    index = np.zeros((min_N1_N2, 2), dtype=int)
    if min_N1_N2 > 1:
        for k in range(min_N1_N2):
            d2 = np.min(diffmat, axis=0)
            index2 = np.argmin(diffmat, axis=0)
            index1 = np.argmin(d2)
            index2 = index2[index1]
            index[k, :] = [index1, index2]
            diffmat[index2, :] = float('inf')
            diffmat[:, index1] = float('inf')
        d = np.mean(np.arccos(np.cos(x1[:, index[:, 0]] - x2[:, index[:, 1]])))
    else:
        d = np.min(diffmat)
        index = np.argmin(diffmat)
        if N1 == 1:
            index = np.array([1, index])
        else:
            index = np.array([index, 1])
    return d, index
