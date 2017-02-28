import sys
import numpy as np
import matplotlib.pyplot as plt

import browserinterface
import algorithms as rt

"""Select appropriate sampling frequency"""
try:
    import json
    with open('./hardware_config.json', 'r') as config_file:
        config = json.load(config_file)
        config_file.close()
    sampling_freq = config['sampling_frequency']
except:
    # default when no hw config file is present
    sampling_freq = 44100


"""capture parameters"""
buffer_size = 4096
num_channels = 2

"""Defining callback"""
def example_callback(audio):

    # play back audio
    browserinterface.send_audio(audio)

browserinterface.register_handle_data(example_callback)


"""START"""
browserinterface.change_config(buffer_frames=buffer_size, 
    channels=num_channels, rate=sampling_freq, volume=80)
browserinterface.start()
browserinterface.loop_callbacks()
