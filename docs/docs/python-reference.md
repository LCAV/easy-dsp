# Intro

This part explains how to get the audio streams in python code and how to send results to the webapp, so it can display them in live.

## How it works

When your write code in the webapp editor and click "Execute", it will be executed as a new python script.
So either you use the browser or write directly a python script that you launch from your terminal, the working is similar.

To easily access the audio streams, and display something in the webapp, we provide a python module `browserinterface`.

The minimum python code is the following:

```python
# First you import the module
import browserinterface

# Then you define the function that will perform some algorithm on the audio streams
def handle(buffer):
    print "Buffer received", len(buffer)

# If you run a python script directly from your terminal, you need to set `standalone` to True
# If you write this code directly in the webapp, you must remove this line (or set `standalone` to False)
browserinterface.standalone = True

# Finally you register your function, so browserinterface will call it every time a new audio buffer is received,
browserinterface.register_handle_data(handle)
# And you start
browserinterface.start()
```

In the following, when we talk about functions and variables, they all come from the module `browserinterface`, so you must prefixe them with `browserinterface.`.

## Reading the configuration

Four variables contain the configuration:

* `rate`: the rate in bits/second;
* `channels`: the number of channels;
* `buffer_frames`: the number of audio frames contained in one buffer;
* `volume`: the volume between 0 and 100.

These are read-only, and you must not change them!

## Receiving the audio streams

You can define a function that will be called each time a new audio buffer is received, by registering it:
```python
def my_function(buffer):
    print "Buffer received", len(buffer)
browserinterface.register_handle_data(my_function)
```

The parameter `buffer` will contain an array of size `buffer_frames` containing arrays of size `channels` containing integers between -32 767 and +32 767.

An example with 5 frames per buffer and 2 channels:

```python
[
  [100, 300],
  [80, 240],
  [130, 0],
  [-800, 123],
  [-400, 0]
]
```

## Using the data handlers

After you performed some algorithms on the audio streams, you may want to display some outputs, like charts, histograms or new audio streams.
What you have to do is to send the data you want to display to a **data handler** of the webapp.
You have two simple steps to do:

1. You create a new data handler using the function `add_handler(name, type, parameters)` which returns an object representing this new instance;
2. You send data to this instance using its method `send_data(data)`.

Once you call the function `add_handler`, a new tab will be created in the webapp, with the name `name` you specified, and the chart/plot/audio player will appear inside.
You can use the part [Data Handlers](data-handlers.md) to see which *types* of data handlers exist, which parameters are supported, and which structure the `data` you send must follow.
