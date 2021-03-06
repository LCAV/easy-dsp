# Intro

Here you will find the different messages that can be exchanged between the different components, inside the UNIX Sockets or the WebSockets.
The titles are always server <> client.

## Main Daemon

### Main Daemon > WSAudio

This connection is only one-way: the main daemon sends messages to WSAudio.

#### Connection

* Socket Type: UNIX Socket;
* File: `/tmp/micros-audio.socket`;
* Transport protocol: TCP.

#### Messages

* Audio configuration: this message allows WSAudio to know the audio configuration choosed, so that it can allocate the correct buffer size.
    * Length in **bytes**: `4*sizeof(int)`;
    * Payload: four integer: `{buffer_frames}{rate}{channels}{volume}`:
        * `buffer_frames`: number of audio frames in one buffer;
        * `rate`: audio rate (in bits/second);
        * `channels`: number of channels;
        * `volume`: ALSA volume of all microphones, between 0 and 100.
    * So the size of the audio buffer is `buffer_size = buffer_frames*channels*sizeof(SND_PCM_FORMAT_S16_LE) / 8` in **bytes** (for now, one audio frame is encoded with a 16-bits little-endian integer).
* Audio buffer: this message contains new audio data.
    * Length: the previous computed `buffer_size`;
    * Payload: `buffer_frames*channels` 16-bits little-endian integer, in the following order (for example with 2 channels):
        1. frames[0].channels[0];
        2. frames[0].channels[1];
        3. frames[1].channels[0];
        4. frames[1].channels[1];
        5. frames[2].channels[0];
        6. frames[2].channels[1];
        7. ...

To differentiate the two messages types, WSAudio only uses the length of the message.
If it's four integers then it is a configuration information, else it is audio data.


### Main Daemon < WSConfig

This connection is only one-way: WSConfig sends messages to the main daemon.

#### Connection

* Socket Type: UNIX Socket;
* File: `/tmp/micros-config.socket`;
* Transport protocol: TCP.

#### Messages

* Audio configuration: this message allows WSConfig to send a new audio configuration to the main daemon.
    * Length in **bytes**: `4*sizeof(int)`;
    * Payload: four integer: `{buffer_frames}{rate}{channels}{volume}`:
        * `buffer_frames`: number of audio frames in one buffer;
        * `rate`: audio rate (in bits/second);
        * `channels`: number of channels;
        * `volume`: ALSA volume of all microphones, between 0 and 100.


## WSAudio > Client

This connection is only one-way: WSAudio sends messages to the webapp or the Python program.

#### Connection

* Protocol: WebSocket;
* Port: 7321.

#### Messages

* Audio configuration: this message allows the client to know the audio configuration chosen.
    * Message type: text;
    * Message format: JSON:

            {
              "buffer_frames": (integer), // the number of frames in one buffer
              "rate": (integer), // the bitrate in bits/second
              "channels": (integer), // the number of channels
              "volume": (integer) // the volume of all microphones, between 0 and 100
            }

    * So the size of the audio buffer is `m.buffer_frames * m.channels * 16 / 8` in **bytes** (16 because a frame is encoded using a 16-bits little-endian integer), where `m` is the received JSON message.

* Audio buffer: this message contains new audio data.
    * Message type: binary;
    * Payload: `m.buffer_frames * m.channels` 16-bits little-endian integer, in the following order (for example with 2 channels):
        1. frames[0].channels[0];
        2. frames[0].channels[1];
        3. frames[1].channels[0];
        4. frames[1].channels[1];
        5. frames[2].channels[0];
        6. frames[2].channels[1];
        7. ...

## WSConfig < Webapp

This connection is only one-way: the webapp sends new audio configuration to WSConfig

#### Connection

* Protocol: WebSocket;
* Port: 7322.

#### Messages

* Audio configuration: this message allows the webapp to send a new audio configuration to WSConfig.
    * Message type: text;
    * Message format: JSON:

            {
              "buffer_frames": (integer), // the number of frames in one buffer
              "rate": (integer), // the bitrate in bits/second
              "channels": (integer), // the number of channels
              "volume": (integer) // the volume of all microphones, between 0 and 100
            }

## Python Daemon <> Webapp

This conenction is two-ways: the webapp can send new Python code to execute, and the daemon can send back the status of the program.

#### Connection (server: Python Daemon)

* Protocol: WebSocket;
* Port: 7320.

#### Messages: Webapp > Python Daemon

* IP address of the board.
    * Message type: text;
    * Message format: JSON: `{"board": (string)}`

* Python code to execute: this message contains new Python code to insert into `base-program.py` and to execute.
    * Message type: text;
    * Payload: just the Python code to execute.

* Interruption of the running code: this messages asks the daemon to stop the current Python code running (each client can only have one Python program running at each time).
    * Message type: text;
    * Payload: just `STOP`.

#### Messages: Python Daemon > Webapp

* Port information: this message indicates to the webapp on which port the new Python program will listen for its WebSocket.
    * Message type: text;
    * Message format: JSON: `{"port": (integer)}`.

* Stdout new line: this message is sent to the webapp each time the new Python program outputs a line on stdout.
    * Message type: text;
    * Message format: JSON: `{"line": (string)}`.

* Stderr new line: this message is sent to the webapp each time the new Python program outputs a line on stderr.
    * Message type: text;
    * Message format: JSON: `{"error": (string)}`.

* Line inserted: this message indicates to the webapp on which line of `base-program.py` the Python code has been inserted (it is usefull to find the correspondance between an error and the original line).
    * Message type: text;
    * Message format: JSON: `{"codeLine": (integer)}`.

* End of the Python program: this message is sent to the webapp when the new Python program exits, with the code returned.
    * Message type: text;
    * Message format: JSON: `{"status": "end", "code": (integer)}`.

* New script: this message indicates to the webapp that a Python script is running and would like to use the webapp for display, and specify on which port the Python program listens for its WebSocket.
    * Message type: text;
    * Message format: JSON: `{"script": (integer)}`.


## Python Daemon <> final-program.py

#### Connection (server: Python Daemon)

* Protocol: WebSocket;
* Port: 7320.

#### Messages: Python Daemon > final-program.py

* IP address of the board.
    * Message type: text;
    * Message format: `(string)`

#### Messages: final-program.py > Python Daemon

* New script: this message indicates to the Python Daemon that a Python script is running and would like to use the webapp for display, and specify on which port the Python program listens for its WebSocket. The Python Daemon will then inform the browser.
    * Message type: text;
    * Message format: JSON: `{"script": (string)}`.


## final-program.py > Webapp

This connection is one-way only: the new Python program can send various outputs to the webapp.

#### Connection

* Protocol: WebSocket;
* Port: just over 7320 (choosen and specified by the Python daemon).

#### Messages

* Audio data to play: this message contains a new audio buffer. The Python program can for example perform something on the audio stream and outputs a new audio stream it wants the webapp to play.
    * Message type: binary;
    * Payload: for now, the configuration must be the same as the input stream: `input_conf.buffer_frames * input_conf.channels` 16-bits little-endian integer, in the following order (for example with 2 channels):
        1. frames[0].channels[0];
        2. frames[0].channels[1];
        3. frames[1].channels[0];
        4. frames[1].channels[1];
        5. frames[2].channels[0];
        6. frames[2].channels[1];
        7. ...

* Audio latency: this message contains the delay in milliseconds between the processing and the reality. We just measure how much time elapsed between the first audio frame we received, and the audio duration we received.
    * Message type: text;
    * Message format: JSON:

            {
                "latency": (float)
            }

* Creation of a new data handler: this message asks the webapp to create a new data handler, that will be then filled with new data.
    * Message type: text;
    * Message format: JSON:

            {
                "addHandler": (string), // name of the handler for display
                "id": (integer), // id chosen to identify the handler. Must be unique
                "type": (string), // type of handler
                "parameters": (object) // optional parameters
            }

* New data for a data handler: this message contains new data for an existing data handler.
    * Message type: text;
    * Message format: JSON:

            {
                "dataHandler": (integer), // id of the existing data handler
                "data": (object) // data for the existing data handler
            }
