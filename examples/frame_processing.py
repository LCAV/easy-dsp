"""
Apply processing frame by frame in the STFT domain (overlapping windows).
"""
import sys
import numpy as np

sys.path.append('..')
import browserinterface
import realtimeaudio as rt

"""Board Parameters"""
buffer_size = 8192
num_windows = 3
sampling_freq = 48000

"""STFT Parameters"""
num_samples = 4096 # determines frequency resolution
num_windows = int(float(buffer_size)/num_samples*2-1)

""" Visualization parameters """
under = 100 # undersample otherwise too many points
num_sec = 5

def init(buffer_frames, rate, channels, volume):
    global stft

    # parameters (block size, number of signals, hop size)
    hop = int(np.floor(num_samples/2))

    # filter (moving average --> low pass) and necessary zero padding
    filter_length = 50
    h = rt.windows.rect(filter_length)/(filter_length*1.)
    zf = filter_length/2
    zb = filter_length/2

    # create STFT object
    stft = rt.transforms.STFT(num_samples, rate, hop,zf=zf,zb=zb,h=h)

def visualize_spectrum(handle):
    global stft

    freq = np.linspace(0,sampling_freq/2,len(stft.X))
    freq = np.ceil(freq).tolist()
    spectrum = np.floor(20. * np.log10( np.maximum( 1e-5, np.abs( stft.X ) ) )).tolist()
    sig = []

    #sig.append({'x': freq, 'y': np.floor(abs(stft.X)).tolist()[::5]})
    lim = 333
    sig.append({'x': freq[:lim], 'y': spectrum[:lim] })
    handle.send_data({'replace': sig})

frame_num = 0
def handle_data(audio):
    global stft, frame_num

    if (browserinterface.buffer_frames != buffer_size 
        or browserinterface.channels != 2):
        print("Did not receive expected audio!")
        return

    # apply stft and istft for a few windows
    for i in range(num_windows):

        # perform analysis and visualize the frame
        stft.analysis(audio[stft.hop*i:stft.hop*(i+1),0])
        visualize_spectrum(c_magnitude)

        # apply filtering and visualize results
        stft.process()
        visualize_spectrum(c_magnitude_f)

    # time plot
    sig = np.array(audio[:,0], dtype=np.float32)/20000
    t = np.linspace(frame_num*buffer_size,(frame_num+1)*buffer_size,buffer_size)/float(sampling_freq)
    sig = {'x': t[::under].tolist(), 'y': sig[::under].tolist()}
    time_plot.send_data({'add':[sig]})
    frame_num += 1

"""Interface functions"""
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(handle_data)
time_plot = browserinterface.add_handler("Time domain", 'base:graph:line', {'xName': 'Duration', 'min': -1, 'max': 1, 'xLimitNb': (sampling_freq/under*num_sec), 'series': [{'name': 'Signal', 'color': 'blue'}]})
c_magnitude = browserinterface.add_handler("Frequency Magnitude", 'base:graph:line', {'min': 0, 'max': 100000, 'xName': 'Frequency', 'series': [{'name': '1'}, {'name': '2'}]})
c_magnitude_f = browserinterface.add_handler("Frequency Magnitude (Filtered)", 'base:graph:line', {'min': 0, 'max': 100000, 'xName': 'Frequency', 'series': [{'name': '1'}, {'name': '2'}]})


"""START"""
browserinterface.change_config(channels=2, buffer_frames=buffer_size, 
    volume=80, rate=sampling_freq)
browserinterface.start()
browserinterface.loop_callbacks()
