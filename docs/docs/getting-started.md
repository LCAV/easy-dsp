# Getting Started

You will find explanations about how to use the project (how to start, the different commands available), and how to develop on it (the global structure, the different components, their interactions and details about the C and Javascript code).

## On the board

### Prerequisites

* [libwebsock](https://github.com/payden/libwebsock);
* [Jansson](http://www.digip.org/jansson/).

### Compilation

* `gcc -o alsa-record-example -lasound alsa-record-example.c`
* `gcc -g -O2 -o client-control client-control.c -lwebsock -ljansso`
* `gcc -g -O2 -o client client.c -lwebsock`

### Launch

Two possibilities: `./demo.sh` (recommanded) or:

    ./alsa-record-example &
    export LD_LIBRARY_PATH=/usr/local/lib
    ./client &
    ./client-control &

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

1. You first have to laund the python daemon: `python code-server.py`;
2. Then you just have to open the file `client.html` in your browser.
