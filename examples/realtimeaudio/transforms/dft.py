# Author: Eric Bezzam
# Date: Jan 31, 2016

"""Methods for using the Discrete Fourier Transform (DFT) for the analysis 
    and synthesis of real signals."""

import numpy as np
from numpy.fft import rfft
from numpy.fft import irfft
import warnings
import utils

try:
    import matplotlib as mpl
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

try:
    import pyfftw
    pyfftw_available = True
except ImportError:
    pyfftw_available = False

try:
    import mkl_fft  # https://github.com/LCAV/mkl_fft
    mkl_available = True
except ImportError:
    mkl_available = False

if matplotlib_available:
    import matplotlib.pyplot as plt

class DFT(object):
    """

    Parent class for performing Fourier Analysis of real signals through the Discrete Fourier Transform (DFT).

    :param nfft: FFT size.
    :type nfft: int
    :param D: Number of signals. Default is 1.
    :type D: int
    :param fs: Sampling frequency.
    :type fs: float
    :param analysis_window: Window to be applied before DFT.
    :type analysis_window: numpy array
    :param synthesis_window: Window to be applied after inverse DFT.
    :type synthesis_window: numpy array
    :param transform: 'fftw' or 'numpy' to use the appropriate library..
    :type transform: str

    """

    def __init__(self, nfft, D=1, fs=1.0, analysis_window=None, synthesis_window=None, transform='numpy'):

        self.nfft = nfft
        self.D = D
        self.analysis_window = analysis_window
        self.synthesis_window = synthesis_window
        self.nbin = int(self.nfft/2+1)
        self.fs = float(fs)
        self.freq = np.linspace(0,self.fs/2,self.nbin)
        self.X = np.zeros((self.nbin,self.D),dtype='complex64')

        if transform == 'fftw':
            if pyfftw_available:
                import pyfftw
                self.transform = transform
                # allocate input and output (assuming real input) for pyfftw
                if self.D==1:
                    self.a = pyfftw.empty_aligned(self.nfft, dtype='float32')
                    self.b = pyfftw.empty_aligned(self.nbin, dtype='complex64')
                    self.c = pyfftw.empty_aligned(self.nfft, dtype='float32')
                    self.forward = pyfftw.FFTW(self.a, self.b)
                    self.backward = pyfftw.FFTW(self.b, self.c, 
                        direction='FFTW_BACKWARD')
                else:
                    self.a = pyfftw.empty_aligned([self.nfft,self.D], dtype='float32')
                    self.b = pyfftw.empty_aligned([self.nbin,self.D], dtype='complex64')
                    self.c = pyfftw.empty_aligned([self.nfft,self.D], dtype='float32')
                    self.forward = pyfftw.FFTW(self.a, self.b, axes=(0, ))
                    self.backward = pyfftw.FFTW(self.b, self.c, axes=(0, ), 
                        direction='FFTW_BACKWARD')
            else: 
                warnings.warn("Could not import pyfftw wrapper for fftw functions. Using numpy's rfft instead.")
                self.transform = 'numpy'
        elif transform == 'mkl':
            if mkl_available: 
                import mkl_fft
            else:
                warnings.warn("Could not import mkl wrapper. Using numpy's rfft instead.")
                self.transform = 'numpy'
        else:
            self.transform = 'numpy'


    def analysis(self, x, plot_spec=False):
        """
        Perform frequency analysis of a real input using DFT.

        :param x: Real input signal in time domain. Must be of size (N,D).
        :type x: numpy array (float32)
        :param plot_spec: Whether or not to plot frequency magnitude spectrum.
        :type plot_spec: bool
        :rtype: Frequency spectrum, numpy array of size (N/2+1,D) and of type complex64.
        """

        # check for valid input
        if self.D==1 and x.size!=self.nfft:
            raise ValueError('Invalid input dimensions.')
        if self.D!=1 and x.shape!=(self.nfft,self.D):
            raise ValueError('Invalid input dimensions.')
        # apply window if needed
        if self.analysis_window is not None:
            x = np.multiply(self.analysis_window, x)
        # apply DFT
        if self.transform == 'fftw':
            self.a[:,] = x
            self.X = self.forward()
        elif self.transform == 'mkl':
            self.X = mkl_fft.rfft(x,axis=0)
        else:
            self.X = rfft(x,axis=0)
        # plot
        if plot_spec:
            utils.plot_spec(self.X,fs=self.fs)
        return self.X

    def synthesis(self, X=None, plot_time=False):
        """
        Perform time synthesis of frequency domain to real signal using the 
        inverse DFT.

        :param X: Real input signal in time domain. Must be of size (N/2+1,D). Default is previously computed DFT.
        :type X: numpy array (complex64)
        :param plot_time: Whether or not to plot time waveform.
        :type plot_time: bool
        :rtype: Time domain signal, numpy array of size (N,D) and of type float32.
        """

        # check for valid input
        if X is not None:
            if self.D==1 and X.size!=self.nbin:
                raise ValueError('Invalid input dimensions.')
            if self.D!=1 and X.shape!=(self.nbin, self.D):
                raise ValueError('Invalid input dimensions.')
            self.X = X
        # inverse DFT
        if self.transform == 'fftw':
            self.b[:] = self.X
            x = self.backward()
        elif self.transform == 'mkl':
            self.X = mkl_fft.irfft(x,axis=0)
        else:
            x = irfft(self.X, axis=0)
        # apply window if needed
        if self.synthesis_window is not None:
            x = np.multiply(self.synthesis_window, x)
        # plot
        if plot_time:
            utils.plot_time(x,fs=self.fs,title='Waveforms computed from DFT then IDFT')
        return x
