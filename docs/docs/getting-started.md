# Getting Started

You will find explanations about how to use the project (how to start, the different commands available), and how to develop on it (the global structure, the different components, their interactions and details about the C and Javascript code).

## On the board

### Prerequisites

* [libwebsock](https://github.com/payden/libwebsock);
* [Jansson](http://www.digip.org/jansson/).

### Compilation

`make` should be enough to compile the three programs.

### Launch

Two possibilities: `./start.sh log-file.txt` (recommended) or:

    ./browser-main-daemon &
    export LD_LIBRARY_PATH=/usr/local/lib
    ./browser-wsaudio &
    ./browser-wsconfig &

To stop it, kill the three programs (or use `./stop.sh`).

## On the computer

### Prerequisites

* Install [ws4py](https://ws4py.readthedocs.io/en/latest/);
* Clone [Rickshaw](https://github.com/shutterstock/rickshaw) in the directory of the `client.html` file: `git clone https://github.com/shutterstock/rickshaw.git`.


### Configuration

The IP address of the board must be specified in two files:

* Lines 8 and 9 of `js/main.js`;
* At the end of `base-program.py`.

### Launch

1. You first have to launch the python daemon: `python code-server.py`;
2. Then you just have to open the file `client.html` in your browser.
