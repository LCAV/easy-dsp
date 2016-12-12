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
* Numpy.


### Configuration

The IP address of the board must be specified in two files:

* Line 1 of `js/main.js`;
* At the end of `base-program.py`.

### Launch

1. You first have to launch the python daemon: `python code-server.py`;
2. Then you just have to open the file `client.html` in your browser;
3. Finally you can write code:
    - You can write code directly in the browser, where a basic example is provided;
    - Or you can write a python script with your favorite editor and launch it like any python script:

            import browserinterface
            import time
            import random

            def my_handle(data):
                print "New buffer", len(data)

            browserinterface.register_handle_hata(my_handle)
            browserinterface.inform_browser = True
            browserinterface.start()

            print "Hello World!"

            c1 = browserinterface.add_handler("First chart", 'base:graph:line', {'xName': 'ok', 'series': ['yNom', 'ynom 22']})
            c2 = browserinterface.add_handler("Polar", 'base:polar:area', {'title': 'Direction', 'series': ['Intensity'], 'legend': {'from': 0, 'to': 360, 'step': 10}})

            c1.send_data([{'x': 1, 'y': 89}, {'x': 1, 'y': 39}])
            c1.send_data([{'x': 2, 'y': 70}, {'x': 2, 'y': 20}])
            c1.send_data([{'x': 3, 'y': 40}, {'x': 3, 'y': -2}])
            c1.send_data([{'x': 4, 'y': 2}, {'x': 4, 'y': 4}])

            for i in range(5, 40):
              c1.send_data([{'x': i, 'y': 20+i*5*random.random()}, {'x': i, 'y': i*5*random.random()}])
              c2.send_data([{'append': (200+i*3)*10}])
              time.sleep(1)

            browserinterface.loop_callbacks()
