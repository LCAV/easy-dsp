import numpy as np
import sys

sys.path.append('..')
import browserinterface
import realtimeaudio as rt

from math import ceil


""" Board parameters """
buffer_size = 4096
sampling_freq = 44100
# sampling_freq = 48000

""" Visualization parameters """
under = 100  # undersample otherwise too many points
num_sec = 5

"""Check for LED Ring"""
try:
    from neopixels import NeoPixels
    import matplotlib.cm as cm
    led_ring = NeoPixels(usb_port='/dev/cu.usbmodem1411',
        colormap=cm.afmhot)
    print("LED ring ready to use!")
except:
    print("No LED ring available...")
    led_ring = False


def when_config(buffer_frames, rate, channels, volume):
    global dft, vad

    dft = rt.transforms.DFT(nfft=buffer_frames)
    vad = rt.VAD(buffer_size, rate, 10, 40e-3, 3, 1.2)

# perform VAD
frame_num = 0
def apply_vad(buffer):
    global dft, vad, frame_num

    if (browserinterface.buffer_frames != buffer_size 
        or browserinterface.channels != 2):
        print("Did not receive expected audio!")
        return

    # only one channel needed
    sig = buffer[:,0]
    X = dft.analysis(sig)
    decision = vad.decision(X)
    # decision = vad.decision_energy(sig, 4)

    # visualization
    t = np.linspace(frame_num*buffer_size,(frame_num+1)*buffer_size,buffer_size)/float(sampling_freq)
    s1 = {'x': [0,0], 'y': [0,0]}
    s2 = {'x': t[::under].tolist(), 'y': sig[::under].tolist()}

    if decision:
        # first is voiced, second is unvoiced
        chart.send_data({'add':[s2, s1]})
        if led_ring: led_ring.lightify_mono(rgb=[0,255,0],realtime=True)
    else:
        chart.send_data({'add':[s1, s2]})
        if led_ring: led_ring.lightify_mono(rgb=[255,0,0],realtime=True)
    frame_num += 1


"""Interface features"""
browserinterface.register_when_new_config(when_config)
browserinterface.register_handle_data(apply_vad)
chart = browserinterface.add_handler("Speech Detection", 'base:graph:line', {'xName': 'Duration', 'min': -10000, 'max': 10000, 'xLimitNb': (sampling_freq/under*num_sec), 'series': [{'name': 'Voice', 'color': 'green'}, {'name': 'Unvoiced', 'color': 'red'}]})

"""START"""
browserinterface.change_config(channels=2, buffer_frames=buffer_size,
    rate=sampling_freq, volume=80)
browserinterface.start()
browserinterface.loop_callbacks()
