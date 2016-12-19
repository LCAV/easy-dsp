# Author: Eric Bezzam
# Date: Feb 10, 2016

import numpy as np
from scipy.signal import fftconvolve
import warnings
import dft

try:
    import matplotlib as mpl
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

if matplotlib_available:
    import matplotlib.pyplot as plt

tol = 1e-14

def fractional_delay(t0, frac_delay=81):
    '''
    Creates a fractional delay filter using a windowed sinc function.
    The length of the filter is fixed by the module wide constant
    `frac_delay_length` (default 81).

    Argument
    --------
    t0: float
    The delay in fraction of sample. Typically between 0 and 1.

    Returns
    -------
    A fractional delay filter with specified delay.
    '''

    return np.hanning(frac_delay)*np.sinc(np.arange(frac_delay)-(frac_delay-1)/2 + t0)

def unit_vec(doa):
    """
    This function takes a 2D (phi) or 3D (phi,theta) polar coordinates
    and returns a unit vector in cartesian coordinates.

    :param doa: (ndarray) An (D-1)-by-N array where D is the dimension and
                N the number of vectors.

    :return: (ndarray) A D-by-N array of unit vectors (each column is a vector)
    """

    if doa.ndim != 1 and doa.ndim != 2:
        raise ValueError("DoA array should be 1D or 2D.")

    doa = np.array(doa)

    if doa.ndim == 0 or doa.ndim == 1:
        return np.array([np.cos(doa), np.sin(doa)])

    elif doa.ndim == 2 and doa.shape[0] == 1:
        return np.array([np.cos(doa[0]), np.sin(doa[0])])

    elif doa.ndim == 2 and doa.shape[0] == 2:
        s = np.sin(doa[1])
        return np.array([s * np.cos(doa[0]), s * np.sin(doa[0]), np.cos(doa[1])])

def gen_far_field_ir(doa, L, fs, c, frac_delay):
    """
    This function generates the impulse responses for all microphones for
    K sources in the far field.

    :param doa: (nd-array) The sources direction of arrivals. This should
                be a (D-1)xK array where D is the dimension (2 or 3) and K
                is the number of sources
    :param R: the locations of the microphones
    :param fs: sampling frequency

    :return ir: (ndarray) A KxMxL array containing all the fractional delay
                filters between each source (axis 0) and microphone (axis 1)
                L is the length of the filter
    """

    M = L.shape[1]
    dim = L.shape[0]
    K = doa.shape[1]
    # the delays are the inner product between unit vectors and mic locations
    delays = np.zeros((K,M),dtype=float)
    ref = L[:,0]
    # for k in range(K):
    #     for mic in range(1,M):
    #         dist = np.array(ref-L[:,mic],ndmin=2)
    #         delays[k,mic] = np.dot(dist,doa[:,k])/c
    delays = -np.dot(doa.T, L) / c
    delays -= delays.min()
    # figure out the maximal length of the impulse responses
    t_max = delays.max()
    D = int(frac_delay+np.ceil(np.abs(t_max * fs)))

    # create all the impulse responses
    fb = np.zeros((K, M, D))
    for k in xrange(K):
        for m in xrange(M):
            t = delays[k, m]
            delay_s = t * fs
            delay_i = int(np.round(delay_s))
            delay_f = delay_s - delay_i
            fb[k, m, delay_i:delay_i+(frac_delay-1)+1] += fractional_delay(delay_f, frac_delay)
    return fb

def compute_snapshot_spec(signals, N, J, hop):
    nbin = N/2+1
    M = signals.shape[1]
    X = np.zeros([M,nbin,J],dtype=complex)
    d = dft.DFT(N,M)
    for j in range(J):
        x, _ = select_slice(signals, j*hop, N)
        X[:,:,j] = d.analysis(x).T
    return X

def select_slice(x, start_sample, num_samples, fs=1.0):
    start_sample = int(start_sample)
    num_samples = int(num_samples)
    end_sample = start_sample + num_samples
    time = np.linspace(start_sample,end_sample,num_samples)/float(fs)
    if x.shape[1]==1:
        return x[start_sample:end_sample], time
    else:
        return x[start_sample:end_sample,:], time

def plot_time(x, fs=1.0, tmin=None, tmax=None, title=None, grid=True, xlabel=None, time=None, samey=False, plt_show=False):

    if matplotlib_available is False:
        warnings.warn("Could not import matplotlib.")
        return
    # plot parameters
    if xlabel is None:
        if fs==1.0:
            xlabel = 'Time [samples]'
        else:
            xlabel = 'Time [s]'
    x_shape = np.shape(x)
    if len(x_shape)==1:
        num_sig = 1
        if time is None:
            time = np.arange(len(x))/float(fs)
        if title is None:
            title = 'Time waveform'
    else:
        num_sig = x_shape[1]
        if time is None:
            time = np.arange(len(x[:,0]))/float(fs)
        if title is None:
            title = 'Time waveforms'
    if tmin is None:
        tmin = min(time)
    if tmax is None:
        tmax = max(time)
    if samey:
        ymax = max([np.amax(x),abs(np.amin(x))])
    # plot
    fig = plt.figure()
    axes = [fig.add_subplot(num_sig,1,i+1) for i in range(num_sig)]
    for i in range(num_sig):
        if i == 0:
            axes[0].set_xlim([tmin,tmax])
            axes[0].set_title(title)
        if num_sig == 1:
            axes[i].plot(time, x)
            ymax = max([np.amax(x),abs(np.amin(x))])
        else:
            axes[i].plot(time,x[:,i])
            axes[i].set_ylabel('#'+str(i+1))
            plt.setp(axes[i].get_yticklabels(), visible=False)
            ymax_i = max([np.amax(x[:,i]),abs(np.amin(x[:,i]))])
        axes[i].grid(grid)
        axes[i].set_xlim([tmin,tmax])
        if samey or num_sig==1:
            axes[i].set_ylim([-ymax,ymax])
        elif not samey and num_sig>1:
            axes[i].set_ylim([-ymax_i,ymax_i])
        if i == num_sig-1:
            axes[i].set_xlabel(xlabel)
        else:
            plt.setp(axes[i].get_xticklabels(), visible=False)
    if plt_show: plt.show()
    return fig

def plot_spec(X, c, fs=1.0, fmin=None, fmax=None, title=None, grid=True, xlabel=None, plt_show=False):
    """ For real signals"""

    if matplotlib_available is False:
        warnings.warn("Could not import matplotlib.")
        return
    # plot parameters
    if xlabel is None:
        if fs==1.0:
            xlabel = 'Normalized frequency'
        else:
            xlabel = 'Frequency [Hz]'
    x_shape = np.shape(X)
    if len(x_shape)==1:
        num_sig = 1
        freq = np.linspace(0,fs/2,len(X))
        if title is None:
            title = 'Magnitude spectrum'
    else:
        num_sig = x_shape[1]
        freq = np.linspace(0,fs/2,len(X[:,0]))
        if title is None:
            title = 'Magnitude spectra'
    if fmin is None:
        fmin = min(freq)
    if fmax is None:
        fmax = max(freq)
    # plot
    # plot
    # print freq, abs(X)
    # plt.figure()
    datay = []
    datax = np.ceil(freq).tolist()[::5]
    to_send = []
    for i in range(num_sig):
        # subplot_id = str(num_sig)+'1'+str(i+1)
        # plt.subplot(int(subplot_id))
        if num_sig == 1:
            to_send.append({'x': datax, 'y': np.floor(abs(X)).tolist()[::5]})
            # plt.plot(freq,abs(X))
        else:
            to_send.append({'x': datax, 'y': np.floor(abs(X[:,i])).tolist()[::5]})
    c.send_data({'replace': to_send})

def compute_delayed_signals(L, angles, signals, SNR=0, fs=48000, c=343., frac_length=81):
    """
    Simulate a source arriving at a particular direction (far-field) by delaying a given signal according to microphone positions.
    """
    # convert given angles to cartesian coordinates
    angles = np.array([angle*np.pi/180. for angle in angles])
    # polar = np.ones((2,len(angles)))
    # polar[1,:] = angles
    doa = unit_vec(angles)

    # Adjust the SNR
    for k in range(len(angles)):
        # We normalize each signal so that \sum E[x_k**2] / E[ n**2 ] = SNR
        s = signals[:,k]
        signals[:,k] = s / np.std(s[np.abs(s) > 1e-2]) * len(angles)
    noise_power = 10 ** (-SNR * 0.1)

    # delays signals appropriately
    signal_length = signals.shape[0]
    impulse_response = gen_far_field_ir(doa, L, fs, c, frac_length)
    clean_recordings = np.zeros((L.shape[1], signal_length + impulse_response.shape[2] - 1), dtype=np.float32)
    for k in range(len(angles)):
        for mic in range(L.shape[1]):
            clean_recordings[mic] += fftconvolve(impulse_response[k, mic], signals[:,k])

    # add noise
    noisy_recordings = clean_recordings + np.sqrt(noise_power) * np.array(np.random.randn(*clean_recordings.shape), dtype=np.float32)
    return noisy_recordings

def buffer_frames(x, N):
    """
    Buffer a vector into frames of length N

    Parameters
    -----------
    x : numpy array (float)
        Samples
    N : int
        length of each frame

    Returns
    -----------
    frames : 2D numpy array
        Matrix where each row is a frame of length N.

    """
    num_frames = int(np.ceil(len(x)/float(N)))
    tmp = np.zeros([N*num_frames,1])
    tmp[0:len(x)] = x
    frames = np.reshape(tmp, [num_frames,N])
    return frames


def spher2cart(spher):
    """
    Convert spherical (radians) to cartesian coodinates.

    Parameters
    -----------
    spher : numpy array (float)
        Spherical coordinates (distance, azimuth, elevation) where each row is a point. Angles in radians.

    Returns
    -----------
    cart : numpy array
        Cartesian coordinates (x,y,z) where each row is a point.

    """

    # obtain coordinates
    if len(spher.shape)==1:
        r = spher[0]
        theta = spher[1]
        phi = spher[2]
    else:
        r = spher[:,0]      # radius
        theta = spher[:,1]  # azimuth
        phi = spher[:,2]    # elevation

    # convert to cartesian
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)

    # store in output
    if len(spher.shape)==1:
        if abs(x) < tol:
            x = 0
        if abs(y) < tol:
            y = 0
        if abs(z) < tol:
            z = 0
        cart = np.array([x, y, z])
    else:
        cart = np.zeros([spher.shape[0], spher.shape[1]])
        cart[:,0] = x
        cart[:,1] = y
        cart[:,2] = z
        for i in range(len(cart[:,0])):
            if abs(cart[i,0]) < tol:
                cart[i,0] = 0
            if abs(cart[i,1]) < tol:
                cart[i,1] = 0
            if abs(cart[i,2]) < tol:
                cart[i,2] = 0

    return cart

def polar2cart(polar):
    """
    Convert polar (radians) to cartesian coodinates.

    Parameters
    -----------
    polar : numpy array (float)
        Polar coordinates (distance, angle) where each column is a point. Angle in radians.

    Returns
    -----------
    cart : numpy array
        Cartesian coordinates (x,y) where each column is a point.

    """

    # obtain coordinates
    if len(polar.shape)==1:
        r = polar[0]
        theta = polar[1]
    else:
        r = polar[0,:]      # radius
        theta = polar[1,:]  # angle

    # convert to cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    if len(polar.shape)==1:
        cart = np.array([x, y])
    else:
        cart = np.zeros([polar.shape[0], polar.shape[1]])
        cart[0,:] = x
        cart[1,:] = y
    cart[cart<tol] = 0
    return cart

def cart2spher(cart):
    """
    Convert cartesian to spherical (radians) coodinates.

    Parameters
    -----------
    cart : numpy array (float)
        Cartesian coordinates (x,y,z) where each row is a point.

    Returns
    -----------
    spher : numpy array
        Spherical coordinates (distance, azimuth, elevation) where each row is a point. Angles in radians.

    """

    if len(cart.shape)==1:
        # obtain coordinates
        x = cart[0]
        y = cart[1]
        z = cart[2]
        # convert to spherical
        r = np.sqrt(x**2 + y**2 + z**2) # distance
        theta = np.arctan2(y,x)         # azimuth
        phi = np.arccos(z/r)            # elevation
        # store in output
        if abs(r) < tol:
            r = 0
        if abs(theta) < tol:
            theta = 0
        if abs(phi) < tol:
            phi = 0
        spher = np.array([r, theta, phi])
    else:
        # obtain coordinates
        x = cart[:,0]
        y = cart[:,1]
        z = cart[:,2]
        # convert to spherical
        r = np.sqrt(x**2 + y**2 + z**2) # distance
        theta = np.arctan2(y,x)         # azimuth
        phi = np.arccos(z/r)            # elevation
        # store in output
        spher = np.zeros([cart.shape[0], cart.shape[1]])
        spher[:,0] = r
        spher[:,1] = theta
        spher[:,2] = phi
        for i in range(len(spher[:,0])):
            if abs(spher[i,0]) < tol:
                spher[i,0] = 0
            if abs(spher[i,1]) < tol:
                spher[i,1] = 0
            if abs(spher[i,2]) < tol:
                spher[i,2] = 0

    return spher


def cart2polar(cart):
    """
    Convert cartesian to polar (radians) coodinates.

    Parameters
    -----------
    cart : numpy array (float)
        Cartesian coordinates (x,y) where each row is a point.

    Returns
    -----------
    polar : numpy array
        Polar coordinates (distance, angle) where each row is a point.

    """

    if len(cart.shape)==1:
        # obtain coordinates
        x = cart[0]
        y = cart[1]
        # convert to polar
        r = np.sqrt(x**2 + y**2)    # distance
        theta = np.arctan2(y,x)
        # store in output
        if abs(r) < tol:
            r = 0
        if abs(theta) < tol:
            theta = 0
        polar = np.array([r, theta])
    else:
        # obtain coordinates
        x = cart[:,0]
        y = cart[:,1]
        # convert to polar
        r = np.sqrt(x**2 + y**2)    # distance
        theta = np.arctan2(y,x)
        # store in output
        polar = np.zeros([cart.shape[0], cart.shape[1]])
        polar[:,0] = r
        polar[:,1] = theta
        for i in range(len(polar[:,0])):
            if abs(polar[i,0]) < tol:
                polar[i,0] = 0
            if abs(polar[i,1]) < tol:
                polar2cart[i,1] = 0

    return polar
