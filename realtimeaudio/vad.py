import numpy as np
from transforms.dft import DFT
from windows import *

try:
    import matplotlib as mpl
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

if matplotlib_available:
    import matplotlib.pyplot as plt

tol = 1e-14
            
class VAD(object):

    """

    Class to apply Voice Activity Detection (VAD) algorithms.

    Methods
    --------
    decision(X)
        Detect voiced segment by estimating noise floor level.
    decision_energy(frame, thresh)
        Detect voiced segment based on energy of the signal.

    """
    
    def __init__(self, N, fs, tau_up=10, tau_down=40e-3, T_up=3, T_down=1.2):

        """
        Constructor for VAD (Voice Activity Detector) class.

        Parameters
        -----------
        N : int
            Length of frame.
        fs : float or int
            Sampling frequency
        tau_up : float
            Time in seconds.
        tau_down : float
            Time in seconds.
        T_up : float
            Time in seconds.
        T_down : float
            Time in seconds.
        """
        
        self.N = N      
        self.T = self.N/float(fs)
        self.tau_up = tau_up
        self.tau_down= tau_down
        self.T_up = T_up
        self.T_down = T_down
        self.fs = fs

        # fmin = 300; fmax = 3400
        # fmin = int(np.floor(float(fmin)/self.fs*self.N))
        # fmax = int(np.floor(float(fmax)/self.fs*self.N))
        # self.W = np.zeros(N/2+1)
        # self.W[fmin:fmax] = 1
        self.W = rect(N/2+1)
        
        self.V = False
        self.L_min = 10
        self.dft = DFT(N)
        
    def decision(self, X):

        """
        Detect voiced segment by estimating noise floor level.

        Parameters
        -----------
        X : numpy array
            RFFT of one signal with length self.N/2+1
        """

        L = np.sqrt(np.sum(self.W*abs(X))**2/len(X))
        
        # estimate noise floor
        if L > self.L_min:
            L_min = (1-self.T/self.tau_up)*self.L_min + self.T/self.tau_up*L
        else:
            L_min = (1-self.T/self.tau_down)*self.L_min + self.T/self.tau_down*L
        # voice activity decision
        if L/L_min < self.T_down:
            V = False
        elif L/L_min > self.T_up:
            V = True
        else:
            V = self.V
            
        self.L_min = L_min
        self.V = V
        
        return V
        
    def decision_energy(self, frame, thresh):

        """
        Detect voiced segment based on energy of the signal.

        Parameters
        -----------
        frame : numpy array
            One signal of length self.N
        thresh : float
            Threshold for detecting voiced segment.
        """
        
        E = np.log10(np.linalg.norm(frame)+tol)
        if E <= thresh:
            return False
        else:
            return True

    def decision_full(self, x):
        dft = DFT(self.N)
        num_frames = int(np.floor(len(x)/float(self.N)))
        plt.figure()
        for k in range(num_frames):
            X = dft.analysis(x[k*self.N:(k+1)*self.N])
            t = np.linspace(k*self.N,(k+1)*self.N,self.N)/float(self.fs)
            decision = self.decision(X)
            if decision and matplotlib_available:
                plt.plot(t,x[k*self.N:(k+1)*self.N],color='g')
            else:
                plt.plot(t,x[k*self.N:(k+1)*self.N],color='r')



