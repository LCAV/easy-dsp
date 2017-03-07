"""
Apply processing frame by frame in the STFT domain.
"""
import numpy as np
from scipy import signal

import browserinterface
import algorithms as rt

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
nrate = sampling_freq/2 # nyquist rate

# # low pass
# numtaps = 50
# cutoff_hz = 200.
# fir_coeff = signal.firwin(numtaps, float(cutoff_hz)/nrate)

# band pass
numtaps = 50
f1, f2 = 2000., 5000.
fir_coeff = signal.firwin(numtaps, [float(f1)/nrate, float(f2)/nrate], 
    pass_zero=False)
    
# ideally pick a buffer size so that length of DFT (buffer_size*2) will be power of two
nfft = 2048
buffer_size = nfft - numtaps + 1
num_channels = 2
transform = 'mkl' # 'numpy', 'mlk', 'fftw'

""" Visualization parameters """
under = 100 # undersample otherwise too many points
num_sec = 5
viz = True   # if false, playback instead


"""Check for LED Ring"""
try:
    import matplotlib.cm as cm
    led_ring = rt.neopixels.NeoPixels(usb_port=led_ring_address,
        colormap=cm.jet, vrange=[70, 100.])
    print("LED ring ready to use!")
except:
    print("No LED ring available...")
    led_ring = False


def log_bands(fmin, fmax, n, fs, nfft):
    ''' Creates logarithmically spaced bands '''

    lfmin = np.log10(fmin)
    lfmax = np.log10(fmax)

    fc = np.logspace(lfmin, lfmax, n + 1, base=10)
    fd = 10**(lfmax - lfmin) / n

    bands_hertz = np.array([ [ f, f * fd ] for f in fc[:-1] ])
    bands = np.array(bands_hertz / fs * nfft, dtype=np.int)

    # clean up
    for i, band in enumerate(bands):
        if band[0] == band[1]:
            band[1] = band[0] + 1
            band[i+1:,:] += 1

    return bands

def bin_spectrum(spectrum, bands, output=None):

    if output is None:
        output = np.zeros(bands.shape[0])

    for b, band in enumerate(bands):
        output[b] = np.mean(spectrum[band[0]:band[1]])

    return output
    

def init(buffer_frames, rate, channels, volume):
    global stft

    # create STFT object - buffer size will be our hop size
    stft = rt.transforms.STFT(buffer_size, rate, hop=buffer_size, 
        transform=transform)
    stft.set_filter(coeff=fir_coeff, zb=numtaps)

def visualize_spectrum(handle, spectrum):
    global stft

    freq = np.linspace(0,sampling_freq/2,len(stft.X))
    freq = np.ceil(freq).tolist()
    spectrum = spectrum.tolist()
    sig = []

    sig.append({'x': freq[::5], 'y': spectrum[::5] })
    handle.send_data({'replace': sig})

if led_ring:
    n_bands = led_ring.num_pixels / 2
    pixel_values = np.zeros(2 * n_bands)
    bands = log_bands(200, 0.9 * sampling_freq / 2, n_bands, sampling_freq, nfft)

def light_spectrum(X):

    if led_ring:

        bin_spectrum(X, bands, output=pixel_values[:n_bands])
        pixel_values[n_bands:] = pixel_values[n_bands-1::-1]
        led_ring.lightify(vals=pixel_values)
    

frame_num = 0
def handle_data(audio):
    global stft, frame_num

    # check for correct audio shape
    if audio.shape != (buffer_size, num_channels):
        print("Did not receive expected audio!")
        return
    
    stft.analysis(audio[:,0])
    spectrum_before = np.floor(20. * np.log10( np.maximum( 1e-5, np.abs( stft.X ) ) ))
    if viz:
        visualize_spectrum(c_magnitude, spectrum_before)
    
    stft.process()
    spectrum_after = np.floor(20. * np.log10( np.maximum( 1e-5, np.abs( stft.X ) ) ))
    if viz:
        visualize_spectrum(c_magnitude_f, spectrum_after)

    # either viz or playback
    if not viz:
        audio[:,0] = stft.synthesis().astype(audio.dtype)
        audio[:,1] = audio[:,0]
        browserinterface.send_audio(audio)

    # time plot
    if viz:
        sig = np.array(audio[:,0], dtype=np.float32)/10000
        t = np.linspace(frame_num*buffer_size,(frame_num+1)*buffer_size,buffer_size)/float(sampling_freq)
        sig = {'x': t[::under].tolist(), 'y': sig[::under].tolist()}
        time_plot.send_data({'add':[sig]})

    light_spectrum(spectrum_before)

    frame_num += 1

"""Interface functions"""
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(handle_data)
if viz:
    time_plot = browserinterface.add_handler("Time domain", 'base:graph:line', {'xName': 'Duration', 'min': -1, 'max': 1, 'xLimitNb': (sampling_freq/under*num_sec), 'series': [{'name': 'Signal', 'color': 'blue'}]})
    c_magnitude = browserinterface.add_handler("Frequency Magnitude", 'base:graph:line', {'min': 0, 'max': 250, 'xName': 'Frequency', 'series': [{'name': '1'}, {'name': '2'}]})
    c_magnitude_f = browserinterface.add_handler("Frequency Magnitude (Filtered)", 'base:graph:line', {'min': 0, 'max': 250, 'xName': 'Frequency', 'series': [{'name': '1'}, {'name': '2'}]})


"""START"""
browserinterface.start()
browserinterface.change_config(channels=num_channels, buffer_frames=buffer_size, volume=80, rate=sampling_freq)
browserinterface.loop_callbacks()
