# Getting Started

## On the board

### Installation

#### Prerequisites

* [libwebsock](https://github.com/payden/libwebsock);
* [Jansson](http://www.digip.org/jansson/);
* Apache and PHP: `sudo apt-get install apache2 libapache2-mod-php5 php5 php5-common`.

#### Setup

* Clone the repository in `/var/bbb-sta321mp`: `cd /var && git clone https://git.epfl.ch/repo/bbb-sta321mp.git`;
* Create a file `/var/bbb-sta321mp/browser-interface/logs.txt` and set the owner to `www-data`: `cd /var/bbb-sta321mp/browser-interface/ && touch logs.txt && chown www-data:www-data logs.txt`;
* Copy the virtualhost configuration file and enable it: `cp /var/bbb-sta321mp/browser-interface/microphones.virtualhost /etc/apache2/sites-available/microphones && a2ensite microphones`;
* Compile the C daemons: `cd /var/bbb-sta321mp/browser-interface/ && make`

## On the computer

### Prerequisites

* Install [ws4py](https://ws4py.readthedocs.io/en/latest/);
* Numpy.


### Launch

1. You first have to launch the Python daemon on your computer: `python code-server.py`;
2. Then you just have to open your browser and access `http://ip.of.the.board:8081`;
3. Using the buttons on the interface, you can easily start the C daemons on the board;
4. Finally you can write code:
    - You can write code directly in the browser, where basics examples are provided;
    - Or you can write a Python script with your favorite editor and launch it like any Python script:

            import browserinterface
            import random

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
