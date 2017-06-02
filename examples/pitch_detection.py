"""
Apply processing frame by frame in the STFT domain.
"""
from __future__ import division, print_function
import numpy as np
from scipy import signal

import browserinterface
import algorithms as rt
from hsv_map import make_hsv_map

from spectral_grid_search import music_peak
from pitch_estimation import estimate_fundamental

"""
Read hardware config from file
"""
try:
    import json
    with open('./hardware_config.json', 'r') as config_file:
        config = json.load(config_file)
        config_file.close()
    sampling_freq = config['sampling_frequency']
    led_ring_address = config['led_ring_address']
except:
    # default when no hw config file is present
    sampling_freq = 44100
    led_ring_address = '/dev/cu.usbmodem1421'

"""
Filter design : 
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html 
"""
nrate = sampling_freq / 2 # nyquist rate

# # low pass
# numtaps = 50
# cutoff_hz = 200.
# fir_coeff = signal.firwin(numtaps, float(cutoff_hz)/nrate)

# ideally pick a buffer size so that length of DFT (buffer_size*2) will be power of two
nfft = 512
buffer_size = nfft
num_channels = 2
transform = 'mkl' # 'numpy', 'mlk', 'fftw'

""" Visualization parameters """
# Colors for low and high intensity in HSV
background = np.array([0.45, 0.2, 0.1])
source = np.array([0.1, 0.8, 0.95])  # yellowish
source = np.array([0.95, 0.8, 0.95])  # 
cmap = make_hsv_map(background, source)


"""Check for LED Ring"""
try:
    import matplotlib.cm as cm
    led_ring = rt.neopixels.NeoPixels(usb_port=led_ring_address,
        colormap=cmap, vrange=[0., 1.])
    print("LED ring ready to use!")
except:
    print("No LED ring available...")
    led_ring = False

n_leds = 60  # number of leds
pitch_range = np.array([100., 10000.])
log_pitch_range = np.log(pitch_range)
bin_range = [int(x) for x in pitch_range / sampling_freq * nfft]
v_range = np.array([10., 300.])  # speed in round-per-second
leds_base = np.exp(-(np.arange(n_leds) - n_leds/2.)**2 / 3.**2)
leds_base /= np.max(leds_base)
leds = np.zeros(n_leds)
dt = float(buffer_size) / sampling_freq
t_prev = 0.

def map_range(x, range1, range2):

    # clamp
    if x < range1[0]:
        x = range1[0]
    elif x > range1[1]:
        x = range1[1]

    # map
    ratio = (x - range1[0]) / (range1[1] - range1[0])
    return ratio * (range2[1] - range2[0]) + range2[0]

def pitch_led_map(pitch):
    global delay, t, t_prev, log_pitch_range, v_range

    speed = map_range(np.log(pitch), log_pitch_range, v_range)
    #speed = map_range(pitch, pitch_range, v_range)

    delay += dt * speed
    delay = delay % n_leds
    delay_i = int(np.floor(delay))
    delay_f = delay - delay_i

    # integer delay
    if delay_i != 0:
        leds[:delay_i] = leds_base[-delay_i:]
        leds[delay_i:] = leds_base[:-delay_i]
    else:
        leds[:] = leds_base[:]

    return leds


def init(buffer_frames, rate, channels, volume):
    global stft

    print('fs=',sampling_freq)

    # create STFT object - buffer size will be our hop size
    stft = rt.transforms.STFT(buffer_size, rate, hop=buffer_size, 
        transform=transform)

pitch = 0.
t = 0.
delay = 0.
frame_num = 0
threshold = 35.

pwr_avg = None
ff_pwr = 0.95
peak_avg = None
ff_peak = 0.9

noise_pixels = np.zeros(n_leds)
noise_ff = 0.8
p_shot_noise = buffer_size / sampling_freq / n_leds * 5

def handle_data(audio):
    global pitch, t, delay, frame_num, threshold, n_leds, pwr_avg, peak_avg, ff_pwr, ff_peak, noise_pixels, noise_ff

    # check for correct audio shape
    if audio.shape != (buffer_size, num_channels):
        print("Did not receive expected audio!")
        return
    
    # STFT processing
    stft.analysis(audio[:,0])
    spectrum = np.floor(20. * np.log10( np.maximum( 1e-5, np.abs( stft.X ) ) ))

    # run MUSIC
    #spectrum = music_peak(audio[:,0], 10, 30, audio.shape[0], nfft)[:nfft//2+1]

    peak_loc = np.argmax(spectrum[bin_range[0]:bin_range[1]+1]) + bin_range[0]

    # leaky average of average power
    if pwr_avg is None:
        pwr_avg = np.mean(spectrum[bin_range[0]:bin_range[1]+1])
    else:
        pwr_avg = ff_pwr * pwr_avg + (1-ff_pwr) * np.mean(spectrum[bin_range[0]:bin_range[1]+1])

    # leaky average of peak power
    if peak_avg is None:
        peak_avg = spectrum[peak_loc]
    else:
        peak_avg = ff_peak * peak_avg + (1-ff_peak) * spectrum[peak_loc]

    # decide if harmonic based on peak to average power ratio
    is_harmonic = False
    if peak_avg - pwr_avg > threshold:
        is_harmonic = True

    # map to light
    if is_harmonic:
        pitch = peak_loc / nfft * sampling_freq
        pixel_values = pitch_led_map(pitch)
    else:
        # lowpass flicker
        fd = np.fft.rfft(np.random.randn(n_leds))
        fd[n_leds//6:] = 0.
        new_flicker = np.fft.irfft(fd)
        flicker = noise_ff * noise_pixels + (1 - noise_ff) * new_flicker
        shot = np.random.rand(n_leds) < p_shot_noise
        pixel_values = flicker * 0.05 + shot * 0.25
    if led_ring:
        led_ring.lightify(vals=pixel_values)

    # count time
    frame_num += 1
    t += buffer_size / sampling_freq

"""Interface functions"""
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(handle_data)

"""START"""
browserinterface.start()
browserinterface.change_config(channels=num_channels, buffer_frames=buffer_size, volume=80, rate=sampling_freq)
browserinterface.loop_callbacks()
