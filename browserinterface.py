#!/usr/local/bin/python

from ws4py.client.threadedclient import WebSocketClient
from wsgiref.simple_server import make_server
from ws4py.websocket import WebSocket
from ws4py.server.wsgirefserver import WSGIServer, WebSocketWSGIRequestHandler
from ws4py.server.wsgiutils import WebSocketWSGIApplication
from threading import Thread
from Queue import Queue
import os
import json
import sys
import socket
import json
import datetime
import numpy as np

# Buffer for audio reception
bi_buffer = 0

# Represent the browser
client = -1

# If the module is called from a python script launched from outside of the browser,
# and which needs the browser to display data, set to True.
# If the python script doesn't need the browser, let it to False.
# If the code is launched from the browser, no need to inform it (it will be already aware), so let it to False.
inform_browser = False

# Messages to send
r_messages = Queue()

# Configuration parameters
rate = -1
channels = 0
buffer_frames = -1
volume = -1

# Callbacks functions
handle_data = 0
when_new_config = 0

def register_handle_data(fn):
    global handle_data
    handle_data = fn

def register_when_new_config(fn):
    global when_new_config
    when_new_config = fn


# Current data handler id
r_id = 0

class DataHandler():
    def __init__(self, id):
        self.id = id

    def send_data(self, data):
        r_messages.put(json.dumps({'dataHandler': self.id, 'data': data}))

# name: name of the graph
# type: type of the graphs: 'base:graph:area', 'base:graph:line', 'base:graph:bar', 'base:graph:scatterplot'
# parameters: parameters of the graph. Typically: {xName: "name of the x axis", seriesNames: ["serie 1", "serie 2", "serie 3"] }
# returns: a Graph object
def add_handler(name, type, parameters):
    global r_id
    r_id = r_id + 1
    r_messages.put(json.dumps({'addHandler': name, 'id': r_id, 'type': type, 'parameters': parameters}))
    return DataHandler(r_id)


# The following variables are used to measure the latency
## Date we began to receive audio stream
bi_audio_start = None
## Number of audio messages we received so far
bi_audio_number = 0

# Audio recordings
bi_recordings = []

# duration in ms
def record_audio(duration, callback):
    global bi_recordings
    bi_recordings.append({'duration': duration, 'callback': callback, 'buffer': np.empty([0, channels], dtype=np.int16), 'ended': False})

# This method goes through all recordings, and send those which must have finished
def handle_recordings():
    global bi_recordings
    for i in reversed(range(len(bi_recordings))):
        recording = bi_recordings[i]
        audio_duration = len(recording['buffer'])*1000/rate
        if audio_duration >= recording['duration']:
            recording['ended'] = True
            recording['callback'](recording['buffer'])
            del bi_recordings[i]


# Connection with WSAudio
class StreamClient(WebSocketClient):
    def received_message(self, m):
        global bi_audio_start
        global bi_audio_number
        global bi_recordings
        global bi_buffer
        if not m.is_binary: # new configuration
            global rate
            global channels
            global buffer_frames
            global volume

            bi_audio_start = None
            bi_audio_number = 0

            m = json.loads(m.data)
            rate = m['rate']
            channels = m['channels']
            buffer_frames = m['buffer_frames']
            volume = m['volume']

            bi_buffer = np.zeros((buffer_frames, channels), dtype=np.int16)
            for recording in bi_recordings:
                recording['buffer'] = np.empty([0, channels], dtype=np.int16)

            if when_new_config != 0:
                when_new_config(buffer_frames, rate, channels, volume)
        else: # new audio data

            # For measuring the latency
            if bi_audio_start == None:
                bi_audio_start = datetime.datetime.now()

            time_diff = datetime.datetime.now() - bi_audio_start
            time_elapsed = time_diff.total_seconds()*1000 # in milliseconds
            audio_received = bi_audio_number*buffer_frames*1000/rate
            audio_delay = time_elapsed - audio_received
            r_messages.put(json.dumps({'latency': audio_delay}))
            bi_audio_number += 1

            # We convert the binary stream into a 2D Numpy array of 16-bits integers
            data = bytearray()
            data.extend(m.data)
            i = 0
            i_frame = 0
            i_channel = 0
            for i in range(len(data) / 2): # we work with 16 bits = 2 bytes
                if data[2*i+1] <= 127:
                    bi_buffer[i_frame][i_channel] = data[2*i] + 256*data[2*i+1]
                else:
                    bi_buffer[i_frame][i_channel] = (data[2*i+1]-128)*256 + data[2*i] - 32768
                i_channel += 1
                if (i % channels) == (channels-1):
                    i_channel = 0
                    i_frame += 1

            # We add the new data to the recordings
            for recording in bi_recordings:
                if not recording['ended']:
                    recording['buffer'] = np.concatenate((recording['buffer'], bi_buffer))
            handle_recordings()

            # We call the potential callback
            if handle_data != 0:
                handle_data(bi_buffer)


# Send a new audio buffer to the browser
def send_audio(buffer):
    global client
    if client != -1:
        try:
            # We need to convert the 2D Numpy array in a binary stream
            nbuffer = []
            for buf in buffer:
                for b in buf:
                    if b >= 0:
                        nbuffer.append(min(b%256, 255))
                        nbuffer.append(min(b/256, 255))
                    else:
                        t = max(0, 32768 + b)
                        nbuffer.append(min(t%256, 255))
                        nbuffer.append(min(t/256 + 128, 255))
            client.send(bytearray(nbuffer), True)
        except socket.error, e:
            print "Error when send_audio: the browser might be disconnected, we remove it"
            client = -1
        except IOError, e:
            print "Error when send_audio: the browser might be disconnected, we remove it"
            client = -1

# Connection with the browser
class WSServer(WebSocket):
    def opened(self):
        global client
        client = self
        while True:
            self.send(r_messages.get(), False)
            r_messages.task_done()

    def close(self, code, reason):
        global client
        global server
        global ws
        sys.stderr.write("The browser disconnected\n")
        client = -1
        os._exit(1)

# WebSocket server, to which the browser can connect
def start_server(port):
    global server
    server = make_server('', port, server_class=WSGIServer,
                         handler_class=WebSocketWSGIRequestHandler,
                         app=WebSocketWSGIApplication(handler_cls=WSServer))
    server.initialize_websockets_manager()
    server.serve_forever()

# In case of standlone, we will informe the python daemon (code-server) that we exist
# so it can forward this information to the browser, which will then connect to us
class PythonDaemonClient(WebSocketClient):
    def opened(self):
        self.send(json.dumps({'script': 9001}))
        self.close()

def inform_browser_query():
    python_daemon = PythonDaemonClient('ws://127.0.0.1:7320/', protocols=['http-only', 'chat'])
    python_daemon.connect()

# Connection to WSAudio
def start_client():
    ws = StreamClient('ws://192.168.1.151:7321/', protocols=['http-only', 'chat'])
    ws.connect()

    ws.run_forever()

def start():
    serverThread = Thread(target = start_server, args = (9001, ))
    serverThread.start()

    if inform_browser:
        inform_browser_query()

    clientThread = Thread(target = start_client)
    clientThread.start()
