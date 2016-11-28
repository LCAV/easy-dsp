# Intro

Here is the global structure of the project, and how the different parts interact.

## Overview

The system is divided in three parts:

1. Daemons in C running on the board;
2. A Python daemon running on the client computer;
3. A webapp executed on the client computer.

These three parts communicate using [WebSockets](https://en.wikipedia.org/wiki/WebSocket) (basically a TCP connection, that you can use easily in Javascript, in browsers).

![Large overview](img/overview-large.svg)

## On the Board

There are three daemons running on the board, all in C:

1. The `main-daemon` is the main part of the system, and the only one interacting with the microphones through ALSA API;
2. The `ws-audio` is a first bridge between the main daemon and a WebSocket, allowing external clients to access the audio streams;
3. The `ws-config` is a second bridge between the main daemon and a WebSocket, allowing external clients to change the configuration of the microphones.

These three daemons communicate together using TCP connections over UNIX sockets.

![Overview board](img/overview-board.svg)

### The Main daemon

This daemon can receive in live the audio streams from the microphones using ALSA API, and can also send a new configuration using the same API.
Then it listens on two UNIX sockets using TCP: `/tmp/micros-audio.socket` and `/tmp/micros-control.socket`.

It sends the audio stream to every client connected to the first socket, and can receive a new audio configuration from the second socket.
The idea is to provide the most basic functionnalities with this daemon, that can be extended using other programs which communicate with it using these two UNIX sockets.

### WSAudio

The WSAudio daemon connects to the main daemon using the UNIX socket `/tmp/micros-audio.socket`, so it receives the audio streams in live.
It also listens on the port 7321 using WebSocket.
It will send the audio streams it receives to every connected client as binary messages.
It doesn't keep in memory the frames: each time it receives new data from the main daemon, it sends it to the connected clients.
When a new client arrives, it starts to receive the audio streams from this moment.

### WSConfig

The WSConfig daemon connects to the main daemon using the UNIX socket `/tmp/micros-control.socket`.
It also listens on the port 7322 using WebSocket.
When a client sends to it a configuration (using a text message containing JSON), it reads the JSON and sends the new information (using an array of integers) to the main daemon.
The main daemon will then interrupt its connection with ALSA, set the new configuration it just received and start again to receive the audio stream.

## On the client

There are two components on the client(s):

1. A python daemon `code-server.py` that listens on a WebSocket and executes the python code it receives on it;
2. The webapp `client.html`, which connects to `ws-audio` and `ws-config` so the user can listen to the audio streams and change the configuration, and also to the local python daemon to which it can send python code written by the user.

### Execution of python code from the browser

In the webapp, there is an editor where the user can write some python code.
When he/she clicks on *Execute* the code is sent to the local python daemon.
Then the python daemon includes the code to an existing python program, `base-program.py` (at the end, it replaces the line `#####INSERT: Here insert code` with the code from the user), and executes it.

This new python program will then connect to `ws-audio`, so it will receive the audio streams in live (and executes the code from the user on it), and it will also listens on a WebSocket (on a port around 7320 specified by the python daemon, which communicated it to the webapp) to which the webapp will connect.
So this new python program will have the possibility to work on the audio streams and to send the output (which can be plots data or a new audio stream) directly to the webapp.

More precisely, the user is invited to write his/her code inside a function `handleData(buffer)` which will be called everytime a new audio buffer is received from the `ws-audio` daemon.

The python daemon will also catch the `stdout` and `stderr` streams and redirect them to the browser, so the user can easily access it.

![Overview client](img/overview-client.svg)


### Execution of a python script

It is also possible to write a python script, that will use the library of the project to easily receive the audio, and to send output to the browser (if wanted).
In that cas the browser is just used as a vizualisation tool.
Just as previously, the script will connect to `ws-audio` and receive in live the audio streams.

However, this time, the outputs are not redirected, so you will be able to see them directly in your console, like any python script.
The library will also choose a port around 7320 by itself, and will communicate it to the python daemon, which will send it to the browser, so this last one can connects to the python script and receive the output (plots or new audio stream).

So can also decide you don't need the browser and just use the connection with the audio stream, and any vizualisation library you want.

![Overview script](img/overview-client-script.svg)
