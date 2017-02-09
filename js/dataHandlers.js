var dataHandlers = new (function() {
  var types = {
    // 'base:graph:plot': handlerOfPlotFunction
  };
  var list = {
    // id: {
    //   type: 'base:graph:plot',
    //   instance: instanceOfTheGraph,
    //   html: $('#graph-id-2')
    // }
  };

  function registerNewType(type, handler) {
    types[type] = handler;
  }
  function init() {
    // For each item in "list", remove the possible html element
    list = {};
  }
  function addHandler(id, type, html, parameters) {
    list[id] = {
      type: type,
      html: html,
      instance: new types[type](html, parameters)
    };
  }
  function addNewData(id, data) {
    list[id].instance.newData(data)
  }

  return {
    registerNewType: registerNewType,
    init: init,
    addHandler: addHandler,
    addNewData: addNewData
  };
})();


function graphRickshaw(renderer) {
  return function graphRickshawLine(html, parameters) {
    var series = [];
    var xName = parameters.xName;
    var xLimitNb = parameters.xLimitNb || -1;
    var xLimitDistance = parameters.xLimitDistance || -1;
    _.forEach(parameters.series, function (serie) {
      series.push({
        data: [],
        color: serie.color || ('rgba(' + Math.floor(Math.random()*255) + ',' + Math.floor(Math.random()*255) + ',' + Math.floor(Math.random()*255) + ',0.75)'),
        name: serie.name
      });
    });
    var graphParams = {
      element: html,
      renderer: renderer,
      height: 250,
      width: 800,
      series: series,
      interpolation: 'linear'
    };
    // yMin and yMax
    if (parameters.min) {
      graphParams.min = parameters.min;
    }
    if (parameters.max) {
      graphParams.max = parameters.max;
    }
    var graph = new Rickshaw.Graph(graphParams);
    graph.render();

    var hoverDetail = new Rickshaw.Graph.HoverDetail({
      graph: graph,
      xFormatter: function(x) { return xName + ": " + x },
      yFormatter: function(y) { return y }
    });

    // Axes
    var xAxis = new Rickshaw.Graph.Axis.X({
      graph: graph
    });
    xAxis.render();

    var yAxis = new Rickshaw.Graph.Axis.Y({
      graph: graph
    });
    yAxis.render();

    function newData(data) {
      var newData = data.add;
      if (data.replace) { // data = {x: [0, 1, 2], y = [[1, 4, 3], [0, -2, 4]]}
        for (var i = 0; i < series.length; i++) {
          series[i].data = [];
        }
        newData = data.replace;
      }
      _.forEach(newData, function (serie, i) {
        if (!serie.x || !serie.y) {
          console.error("x or y attribute is missing for serie", i);
          return;
        }
        if (serie.x.length != serie.y.length) {
          console.error("x and y have different sizes for serie", i);
          return;
        }
        _.forEach(serie.x, function (x, j) {
          series[i].data.push({x: x, y: serie.y[j]});
        });

        if (serie.color) {
          series[i].color = serie.color;
        }
      });

      if (xLimitNb != -1) {
        var xs = _.sortedUniq(_.sortBy(_.flatten(_.map(series, function (serie) {
          return _.map(serie.data, 'x');
        }))));
        var xlimit = xs[xs.length - xLimitNb];
        if (xlimit !== undefined) {
          _.forEach(series, function (serie) {
            serie.data = _.filter(serie.data, function (d) {
              if (d.x < xlimit) {
                return false;
              }
              return true;
            })
          });
        }
      }

      graph.update();
    }

    return {
      newData: newData
    }
  }
}




function plotPolarType(type) {
  return function plotPolar(html, parameters) {
    var data = [];
    _.forEach(parameters.series, function (serie) {
      var d = {
        t: [],
        r: [],
        name: serie,
        geometry: type,
        groupId: 0
      };
      var stepSize = 360.0/parameters.numPoints;
      for (var i = 0.0; i < 360.0+stepSize; i +=stepSize) {
        d.t.push(i);
      }

      data.push(d);
    });

    
    if (parameters.orientation !== undefined) {
      var orient = parameters.orientation;
    }
    else {
      var orient = 0.0
    }
    var c = {
      data: data,
      layout: {
        title: parameters.title || '',
        width: 350,
        height: 350,
        margin: { left: 30, right: 30, top: 30, bottom: 30, pad: 0 },
        angularAxis: { domain: null },
        font: { family: 'Arial, sans-serif', size: 12, color: 'grey' },
        direction: 'counterclockwise',
        orientation: orient,
        barmode: 'stack',
        backgroundColor: 'ghostwhite',
        showLegend: false
      }
    };
    if (parameters.rmax !== undefined) {
      c.layout.radialAxis = {domain: [0.0, parameters.rmax]};
    }
    else {
      c.layout.radialAxis = {domain: [0.0, 1.0]};
    }
    var m = micropolar.Axis();
    m.config(c);
    m.render(d3.select(html));

    function newData(data) {
      _.forEach(data, function (d, i) { // loop on the series
        if (d.append) { // we ca append a new point
          c.data[i].r.push(d.append);
        } else if (d.replace) { // or replace the whole values
          c.data[i].r = d.replace;
        }
      });
      m.config(c);
      m.render(d3.select(html));
    }

    return {
      newData: newData
    };
  };
}

function heatmap(html, parameters) {
  // store the current canvas size [x, y]
  var size = [800, 600];

  var canvas = $('<canvas id="supercanvas" width="' + size[0] + '" height="' + size[1] + '"></canvas>');
  $(html).append(canvas);
  var heat = simpleheat(canvas.get(0));
  if (parameters.max) {
    heat.max(parameters.max);
  }

  heat.radius(2, 0);

  function newData(data) {
    // data = [ [1, 2, 0, 3, 4], [0, 2, 2, 1, 1], [5, 2, 6, 5, 4]]

    // Do we have to resize the canvas?
    if (data.length > 0 && data[0].length > 0 && (data.length != size[0] || data[0].length != size[1])) {
      canvas.attr('height', data.length);
      canvas.attr('width', data[0].length);
      size = [data.length, data[0].length];
      heat.resize();
    }

    var hdata = [];
    _.forEach(data, function (row, i) {
      _.forEach(row, function (v, j) {
        hdata.push([i, j, v]);
      })
    });
    heat.data(hdata);

    if (parameters.min) {
      heat.draw(parameters.min);
    } else {
      heat.draw();
    }
  }

  return {
    newData: newData
  };
}


dataHandlers.registerNewType('base:graph:line', graphRickshaw('line'));
dataHandlers.registerNewType('base:graph:area', graphRickshaw('area'));
dataHandlers.registerNewType('base:graph:bar', graphRickshaw('bar'));
dataHandlers.registerNewType('base:graph:plot', graphRickshaw('scatterplot'));
dataHandlers.registerNewType('base:polar:area', plotPolarType('AreaChart'));
dataHandlers.registerNewType('base:polar:line', plotPolarType('LinePlot'));
dataHandlers.registerNewType('base:heatmap', heatmap);

// dataHandlers.registerNewType('customtype', function (html, parameters) {
//   $(html).append(JSON.stringify(parameters) + '<br />');
//   return {
//     newData: function (data) {
//       $(html).append(JSON.stringify(data) + '<br />');
//     }
//   }
// });
