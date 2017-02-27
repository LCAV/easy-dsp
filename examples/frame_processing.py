"""
Apply processing frame by frame in the STFT domain (overlapping windows).
"""
import sys
import numpy as np
from scipy import signal

sys.path.append('..')
import browserinterface
import realtimeaudio as rt

"""Board Parameters"""
sampling_freq = 44100
# sampling_freq = 48000

"""
Filter design : 
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html 
"""
nrate = sampling_freq/2 # nyquist rate

# low pass
numtaps = 50
cutoff_hz = 200.
fir_coeff = signal.firwin(numtaps, float(cutoff_hz)/nrate)

# # band pass
# numtaps = 50
# f1, f2 = 2000., 5000.
# fir_coeff = signal.firwin(numtaps, [float(f1)/nrate, float(f2)/nrate], 
#     pass_zero=False)
    
# ideally pick a buffer size so that length of DFT (buffer_size*2) will be power of two
buffer_size = 2048-(numtaps/2)

""" Visualization parameters """
under = 100 # undersample otherwise too many points
num_sec = 5
viz = False   # if false, playback instead

def init(buffer_frames, rate, channels, volume):
    global stft

    # create STFT object - buffer size will be our hop size
    stft = rt.transforms.STFT(2*buffer_size, rate, transform="fftw")
    stft.set_filter(coeff=fir_coeff, zb=numtaps/2, zf=numtaps/2)

def visualize_spectrum(handle):
    global stft

    freq = np.linspace(0,sampling_freq/2,len(stft.X))
    freq = np.ceil(freq).tolist()
    spectrum = np.floor(20. * np.log10( np.maximum( 1e-5, np.abs( stft.X ) ) )).tolist()
    sig = []

    sig.append({'x': freq[::5], 'y': spectrum[::5] })
    handle.send_data({'replace': sig})

frame_num = 0
def handle_data(audio):
    global stft, frame_num

    if (browserinterface.buffer_frames != buffer_size 
        or browserinterface.channels != 2):
        print("Did not receive expected audio!")
        return
    
    stft.analysis(audio[:,0])
    if viz:
        visualize_spectrum(c_magnitude)
    
    stft.process()
    if viz:
        visualize_spectrum(c_magnitude_f)

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

    frame_num += 1

"""Interface functions"""
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(handle_data)
if viz:
    time_plot = browserinterface.add_handler("Time domain", 'base:graph:line', {'xName': 'Duration', 'min': -1, 'max': 1, 'xLimitNb': (sampling_freq/under*num_sec), 'series': [{'name': 'Signal', 'color': 'blue'}]})
    c_magnitude = browserinterface.add_handler("Frequency Magnitude", 'base:graph:line', {'min': 0, 'max': 250, 'xName': 'Frequency', 'series': [{'name': '1'}, {'name': '2'}]})
    c_magnitude_f = browserinterface.add_handler("Frequency Magnitude (Filtered)", 'base:graph:line', {'min': 0, 'max': 250, 'xName': 'Frequency', 'series': [{'name': '1'}, {'name': '2'}]})


"""START"""
browserinterface.change_config(channels=2, buffer_frames=buffer_size, 
    volume=80, rate=sampling_freq)
browserinterface.start()
browserinterface.loop_callbacks()
