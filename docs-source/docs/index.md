# Welcome

This project is a browser based interface for prototyping real-time processing algorithms using embedded multi channel audio acquisition platforms.
The interface allows to interact in an easy manner with a GNU/Linux enabled single board computer and to run algorithms in real-time on distant audio streams from a host computer.
It creates a bridge between the user's computer and the audio acquisition board.

More precisely, the interface permit to:

  * change the board configuration;
  * write Python code;
  * execute it in real-time on the audio streams;
  * visualize data (typically outputs of the algorithm).

The project is composed of three main components, communicating together using WebSockets:

  * daemons in C running on the board: they transmit the audio streams and listen for configuration changes;
  * a Python daemon: it executes Python code sent by the interface;
  * the interface itself.

Finally, a Python module helps the user to access the audio streams, the configuration, and to create data visualizations.
This module connects to the C daemons (to receive the streams and the configuration) and to the browser to send data in real-time for visualizations.

The repository of the project is here: [https://github.com/LCAV/easy-dsp](https://github.com/LCAV/easy-dsp).
