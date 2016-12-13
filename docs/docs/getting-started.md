# Getting Started

## On the board

### Prerequisites

* [libwebsock](https://github.com/payden/libwebsock);
* [Jansson](http://www.digip.org/jansson/).

### Compilation

`make` should be enough to compile the three programs `browser-main-daemon`, `browser-wsaudio` and `browser-wsconfig`.

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

* Line 1 of `js/main.js`, the variable `boardIp`;
* At the beginning of `browserinterface.py`, the variable `bi_board_ip`.

### Launch

1. You first have to launch the Python daemon: `python code-server.py`;
2. Then you just have to open the file `client.html` in your browser;
3. Finally you can write code:
    - You can write code directly in the browser, where a basic example is provided;
    - Or you can write a Python script with your favorite editor and launch it like any Python script:

            import browserinterface
            import random

            # If that case, you have to inform the browser of your existence,
            # to be able to use it for display
            browserinterface.inform_browser = True

            print "Simple program"

            # First we define two data handlers: one line chart and one polar chart
            c1 = browserinterface.add_handler("First chart", 'base:graph:line', {'xName': 'Duration', 'xLimitNb': 180, 'series': [{'name': 'Intensity 1'}, {'name':'Intensity 2'}]})
            c2 = browserinterface.add_handler("Polar", 'base:polar:area', {'title': 'Direction', 'series': ['Intensity'], 'legend': {'from': 0, 'to': 360, 'step': 1}})

            c1.send_data({'add': [{'x': [1, 2, 3, 4], 'y': [89, 70, 40, 2, 3]}, {'x': [1, 2, 3, 4], 'y': [39, 20, -2, 4]}]})

            i = 4

            def handle_buffer(buffer):
                # print "New buffer", len(buffer)
                global i
                i += 1
                # We send some random data
                c1.send_data({'add': [{'x': [i], 'y': [20+i*5*random.random()]}, {'x': [i], 'y': [i*5*random.random()]}]})
                c2.send_data([{'append': (200+i*3)*10}])

            # We register this function as a callback function, called every time a new audio buffer is received
            browserinterface.register_handle_data(handle_buffer)


            def new_config_is_here(buffer_frames, rate, channels, volume):
                print "New config received: buffer_frames, rate, channels, volume"
                print buffer_frames, rate, channels, volume

            # We register our function so it will be called when a new configuration arrives
            browserinterface.register_when_new_config(new_config_is_here)

            # We start the module, so it will connect to the daemons to receive the audio stream
            browserinterface.start()

            # This call is blocking and will never return
            # So the code you put below will never be executed
            # It's an infinite loop inside which your callbacks will be called
            browserinterface.loop_callbacks()
