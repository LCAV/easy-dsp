# Definition

A data handler is an object in the webapp which can be used to handle some data.

Typically, they are used to vizualise the output of the program written by the user, which takes into input the audio streams, and could want to draw some charts as a result.
Several data handlers come with the project, but you can also easily write your own, as explained.

There are two main parts when using a data handler:

1. You create it, with some configuration information;
2. You send data to it.

## DataHandler: draw classic charts

This handler can be used to draw line charts, histograms.
There are always 2D charts, with an x axis and an y axis.

<img src="/img/handler-ex1.png" style="height: 100px !important;" />
<img src="/img/handler-ex2.png" style="height: 100px !important;" />
<img src="/img/handler-ex3.png" style="height: 100px !important;" />

### Configuration

The following options are accepted during the creation:

* `xName` (string): name of the x axis;
* `series` (array[string]): names of the different series. **This parameter fixes the number of series to display**;
* `min` (number) [optional]: minimum value for y;
* `max` (number) [optional]: maximum value for y.

### Sending data

#### Adding a point

You can add a new point to each serie:

```json
[
  {"x": 3, "y": 39},
  {"x": 3, "y": -3},
  {"x": 3, "y": 23.1}
]
```

The size of the array must be the number of series (specified during the creation).

#### Replacing all the points

You can also replace all the points of all the series:

```json
{
  "x": [0, 1, 2, 3, 4],
  "y": [
    [12, 4, 1, 1, 5],
    [6, 4, 2, 0, -2],
    [0, 0.5, 1, 0.5, 2]
  ]
}
```

Here we have three series of five points.
Each serie must have the same number of points, matching the length of the `x` array.

## DataHandler: draw polar charts

This handler can draw polar charts (simple ones for now).

<img src="/img/handler-polar-ex1.png" style="height: 200px !important; display: block; margin: 0 auto;" />

### Configuration

The following options are accepted during the creation:

* `title` (string): name of the chart;
* `series` (array[string]): names of the different series. **This parameter fixes the number of series to display**;
* `legend` (object): defines the scale and contains the following parameters:
    * `from` (number): the value corresponding to a start from the north;
    * `to` (number): the value corresponding to the arrival to the north after one revolution;
    * `step` (number): the size of the subdivision.


### Sending data

#### Adding an entry

You can add new data to each serie:

```json
[
  {"append": 10},
  {"append": 4},
  {"append": 43}
]
```

The size of the array must be the number of series (specified during the creation).
The new values will be pushed at the end of previous data of each serie.

## DataHandler: log some data

## Write your own data handler
