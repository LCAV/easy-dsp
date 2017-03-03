# Author: Eric Bezzam
# Date: Feb 1, 2016

"""Class for real-time STFT analysis and processing."""
from __future__ import division

import numpy as np
from dft import DFT
import warnings
import windows
import utils

try:
    import matplotlib as mpl
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

if matplotlib_available:
    import matplotlib.pyplot as plt

class STFT:
    """
    Methods
    --------
    analysis(x_n)
        Perform STFT on most recent samples.
    process(h)
        Perform filtering in frequency domain.
    synthesis()
        Transform to time domain and use overlap-and-add to reconstruct the 
        output.
    set_filter(h, zb, zf)
        Set time-domain filter with appropriate zero-padding.
    get_prev_samples()
        Get previous reconstructed samples.
    zero_pad_front(zf)
        Set zero-padding at beginning of frame.
    zero_pad_back(zb)
        Set zero-padding at end of frame.
    reset()
        Reset state variables. Necessary after changing or setting the filter.
    visualize_frame(fmin=None,fmax=None,plot_time=False)
        Visualize frequency spectrum of current frame.
    spectogram(x, fmin=None,fmax=None,tmin=None,tmax=None,plot_time=False)
        Plot spectrogram according to object's parameters and given signal.
    """
    def __init__(self, N, fs, hop=None, analysis_window=None, 
        synthesis_window=None, channels=1, transform='numpy'):
        """
        Constructor for STFT class.

        Parameters
        -----------
        N : int
            number of samples per frame
        fs : float
            Sampling frequency.
        hop : int
            hop size
        wA : numpy array
            analysis window
        wS : numpy array
            synthesis window
        channels : int
            number of signals
        """
        # initialize parameters
        self.N = N          # number of samples per frame
        self.fs = fs
        self.D = channels         # number of channels
        if hop is not None: # hop size
            self.hop = hop  
        else:
            self.hop = self.N/2
        self.hop = int(np.floor(self.hop))

        # analysis window
        if analysis_window is not None:
            self.analysis_window = analysis_window
        elif analysis_window is None and self.hop ==self.N/2:
            self.analysis_window = windows.sine(self.N)
        else:
            self.analysis_window = None
        # synthesis window
        if synthesis_window is not None:  
            self.synthesis_window = synthesis_window
        elif synthesis_window is None and self.hop ==self.N/2:
            self.synthesis_window = windows.sine(self.N)
        else:
            self.synthesis_window = None

        # create DFT object
        self.transform = transform
        self.nfft = self.N
        self.nbin = self.nfft // 2 + 1
        self.freq = np.linspace(0,self.fs/2,self.nbin)
        self.dft = DFT(nfft=self.nfft,fs=self.fs,num_sig=self.D,
                analysis_window=self.analysis_window,
                synthesis_window=self.synthesis_window,
                transform=self.transform)

        self.fft_out_buffer = np.zeros(self.nbin, dtype=np.complex64)

        # initialize filter + zero padding --> use set_filter
        self.zf = 0; self.zb = 0
        self.H = None       # filter frequency spectrum

        # state variables
        self.num_frames = 0            # number of frames processed so far
        self.n_state = self.N - self.hop

        # allocate all the required buffers
        self.make_buffers()

    def make_buffers(self):

        # The input buffer, float32 for speed!
        self.fft_in_buffer = np.zeros((self.nfft, self.D), dtype=np.float32)
        #  a number of useful views on the input buffer
        self.x_p = self.fft_in_buffer[self.zf:self.zf+self.n_state,:]  # State buffer
        self.fresh_samples = self.fft_in_buffer[self.zf+self.n_state:self.zf+self.n_state+self.hop,:]
        self.old_samples = self.fft_in_buffer[self.zf+self.hop:self.zf+self.hop+self.n_state,:]

        self.y_p = np.zeros((self.zb, self.D), dtype=np.float32)  # prev reconstructed samples
        self.X = np.zeros((self.nbin, self.D), dtype=np.complex64)       # current frame in STFT domain

    def reset(self):
        """
        Reset state variables. Necessary after changing or setting the filter or zero padding.
        """
        self.num_frames = 0
        self.nbin = self.nfft // 2 + 1
        self.freq = np.linspace(0,self.fs/2,self.nbin)

        self.fft_in_buffer[:,:] = 0.
        self.X[:,:] = 0.
        self.y_p[:,:] = 0.

        self.dft = DFT(nfft=self.nfft,fs=self.fs,num_sig=self.D,
            analysis_window=self.analysis_window,
            synthesis_window=self.synthesis_window,
            transform=self.transform)

    def zero_pad_front(self, zf):
        """
        Set zero-padding at beginning of frame.
        """
        self.zf = zf
        self.nfft = self.N+self.zb+self.zf
        if self.analysis_window is not None:
            self.analysis_window = np.concatenate((np.zeros(zf), self.analysis_window))
        if self.synthesis_window is not None:
            self.synthesis_window = np.concatenate((np.zeros(zf), self.synthesis_window))

    def zero_pad_back(self, zb):
        """
        Set zero-padding at end of frame.
        """
        self.zb = zb
        self.nfft = self.N+self.zb+self.zf
        if self.analysis_window is not None:
            self.analysis_window = np.concatenate((self.analysis_window, np.zeros(zb)))
        if self.synthesis_window is not None:
            self.synthesis_window = np.concatenate((self.synthesis_window, np.zeros(zb)))

    def set_filter(self, coeff, zb=None, zf=None, freq=False):
        """
        Set time-domain filter with appropriate zero-padding.

        Frequency spectrum of the filter is computed and set for the object. 
        There is also a check for sufficient zero-padding.

        Parameters
        -----------
        coeff : numpy array 
            Filter in time domain.
        zb : int
            Amount of zero-padding added to back/end of frame.
        zf : int
            Amount of zero-padding added to front/beginning of frame.
        """
        # apply zero-padding
        if zb is not None:
            self.zero_pad_back(zb)
        if zf is not None:
            self.zero_pad_front(zf)  
        self.reset()      
        if not freq:
            # compute filter magnitude and phase spectrum
            self.H = np.complex64(np.fft.rfft(coeff, self.nfft, axis=0))
            # check for sufficient zero-padding
            if self.nfft < (self.N+len(coeff)-1):
                raise ValueError('Insufficient zero-padding for chosen number of samples per frame (L) and filter length (h). Require zero-padding such that new length is at least (L+h-1).')
        else:
            if len(coeff)!=self.nbin:
                raise ValueError('Invalid length for frequency domain coefficients.')
            self.H = coeff

        # We need to reallocate buffers after changing zero padding
        self.make_buffers()


    def analysis(self, x_n):
        """
        Transform new samples to STFT domain for analysis.

        Parameters
        -----------
        x_n : numpy array
            [self.hop] new samples.

        Returns
        -----------
        self.X : numpy array 
            Frequency spectrum of given frame.
        """

        if x_n.ndim == 1:
            x_n = x_n[:,None]

        # check for valid input
        if x_n.shape[0]!=self.hop:
            raise ValueError('Invalid input dimensions.')
        if self.D > 1 and x_n.shape[1]!=self.D:
            raise ValueError('Invalid input dimensions.')

        # form current frame and store for next frame
        self.fresh_samples[:,:] = x_n[:,:]

        # apply DFT to current frame
        self.X = self.dft.analysis(self.fft_in_buffer)
        self.num_frames += 1

        #
        self.x_p[:] = self.old_samples

    def process(self):
        """
        Apply filtering in STFT domain.

        Returns
        -----------
        self.X : numpy array 
            Frequency spectrum of given frame.
        """
        if self.H is None:
            warnings.warn("No filter given to the STFT object.")
        else:
            np.multiply(self.X, self.H, self.X)

    def synthesis(self):
        """
        Transform to time domain and reconstruct output with overlap-and-add.

        Returns
        -----------
        out: numpy array
            Reconstructed array of samples of length [self.hop]
        """
        # apply IDFT to current frame
        y = self.dft.synthesis(self.X)

        # reconstruct output
        out = y[0:self.hop,:]
        out[:self.zb,:] += self.y_p[:,:]

        # update state variables
        self.y_p[:,:] = y[-self.zb:,:]

        return out


    def get_prev_samples(self):
        """
        Get reconstructed previous samples.
        """
        return self.y_p


    def visualize_frame(self, fmin=None,fmax=None,plot_time=False):
        """
        Visualize frequency spectrum of current frame.

        Parameters
        -----------
        fmin : float
            Lower limit for plotting frequency spectrum.
        fmax : float
            Upper limit for plotting frequency spectrum.
        plot_time : bool
            Whether or not to plot corresponding time frame.
        """
        # check if matplotlib imported
        if matplotlib_available is False:
            warnings.warn("Could not import matplotlib.")
            return
        # plot DFT
        time_window = np.arange(2,dtype=float)/self.fs*self.nfft + \
            float(self.num_frames)*self.hop/self.fs - float(self.hop)/self.fs
        title = 'Magnitude spectrum: '+str(time_window[0])[:5]+' s to '+str(time_window[1])[:5]+' s'
        utils.plot_spec(self.X, fs=self.fs, fmin=fmin, fmax=fmax, title=title)
        # plot time waveform
        if plot_time is True:
            y = self.dft.synthesis(self.X)
            time = np.arange(len(y))/float(self.fs) + float(self.num_frames)*self.hop/self.fs - float(self.hop)/self.fs
            title = 'Time waveform: '+str(time_window[0])[:5]+' s to '+str(time_window[1])[:5]+' s'
            utils.plot_time(y, title=title, time=time)

    def spectrogram(self, x, fmin=None,fmax=None,tmin=None,tmax=None, plot_time=False):
        """
        Plot spectrogram according to object's parameters and given signal.

        Parameters
        -----------
        x : numpy array
            Time domain signal.
        fmin : float
            Lower limit for plotting spectrogram.
        fmax : float
            Upper limit for plotting spectrogram.
        tmin : float
            Lower limit for plotting spectrogram.
        tmax : float
            Upper limit for plotting spectrogram.
        plot_time : bool
            Whether or not to plot input waveform.
        """
        if matplotlib_available == False:
            warnings.warn("Could not import matplotlib.")
            return
        self.reset()
        # calculate spectrogram
        num_frames = int(np.floor(len(x)/self.hop))
        dur = float(num_frames*self.hop)/self.fs
        Sx = np.ones([num_frames,self.nbin])
        for i in range(num_frames):
            sig = x[i*self.hop:(i+1)*self.hop]
            X = self.analysis(sig)
            Sx[i,:] = np.log10(np.abs(np.conj(X)*X))
        # plot spectrogram
        f = np.linspace(0,self.fs/2,self.nbin)
        t = np.linspace(0,dur,num=num_frames,dtype=float)
        plt.figure()
        im = plt.pcolormesh(t,f,Sx.T)
        plt.ylabel('Frequency [Hz]')
        if fmin == None:
            fmin = min(f)
        if fmax == None:
            fmax = max(f)
        plt.ylim([fmin,fmax])
        plt.xlabel('Time [sec]')
        if tmin == None:
            tmin = min(t)
        if tmax == None:
            tmax = max(t)
        plt.xlim([tmin,tmax])
        plt.title('Spectrogram')
        plt.colorbar(im, orientation='vertical')
        # plot time waveform
        if plot_time==True:
            utils.plot_time(x[0:num_frames*self.hop],fs=self.fs)
        self.reset()

