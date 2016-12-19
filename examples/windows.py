# Author: Eric Bezzam
# Date: Feb 8, 2016

import numpy as np

def rect(N):
    """Rectangular window function."""
    return np.ones(N, dtype='float32')

def sine(N):
    """Sine window function. Must be of even length."""
    # if (N % 2 == 1):
    #     raise ValueError('Window length must be even.')
    n = np.arange(N, dtype='float32')
    w = np.sin((n+0.5)*np.pi/N)
    return w

def hann(N):
    """Hann window function. Must be of even length."""
    # if (N % 2 == 1):
    #     raise ValueError('Window length must be even.')
    n = np.arange(N, dtype='float32')
    w = 0.5*(1-np.cos(2*np.pi*n/(N)))
    return w