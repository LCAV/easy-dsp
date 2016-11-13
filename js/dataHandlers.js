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
    _.forEach(parameters.series, function (serieName) {
      series.push({
        data: [],
        color: 'rgba(' + Math.floor(Math.random()*255) + ',' + Math.floor(Math.random()*255) + ',' + Math.floor(Math.random()*255) + ',0.75)',
        name: serieName
      });
    })
    var graph = new Rickshaw.Graph({
      element: html,
      renderer: renderer,
      height: 250,
      width: 800,
      series: series
    });
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
      // data = [{x: 23, y: -2}]
      _.forEach(data, function (d, i) {
        series[i].data.push(d);
      });
      graph.update();
    }

    return {
      newData: newData
    }
  }
}

function plotPolar(html, parameters) {
  var data = [];
  _.forEach(parameters.series, function (serie) {
    var d = {
      t: [],
      r: [],
      name: serie,
      geometry: 'AreaChart',
      groupId: 0
    };
    for (var i = parameters.legend.from; i < parameters.legend.to; i += parameters.legend.step) {
      d.t.push(i);
    }
    data.push(d);
  });

  var c = {
    data: data,
    layout: { title: parameters.title || '', width: 350, height: 350, margin: { left: 30, right: 30, top: 30, bottom: 30, pad: 0 },
    angularAxis: { domain: null }, font: { family: 'Arial, sans-serif', size: 12, color: 'grey' },
    direction: 'clockwise', orientation: 270, barmode: 'stack', backgroundColor: 'ghostwhite', showLegend: false }
  };
  var m = micropolar.Axis();
  m.config(c);
  m.render(d3.select(html));

  function newData(data) {
    _.forEach(data, function (d, i) {
      if (d.append) {
        c.data[i].r.push(d.append);
      }
    });
    m.config(c);
    m.render(d3.select(html));
  }

  return {
    newData: newData
  };
}

dataHandlers.registerNewType('base:graph:line', graphRickshaw('line'));
dataHandlers.registerNewType('base:graph:area', graphRickshaw('area'));
dataHandlers.registerNewType('base:graph:bar', graphRickshaw('bar'));
dataHandlers.registerNewType('base:graph:plot', graphRickshaw('scatterplot'));
dataHandlers.registerNewType('base:polar:area', plotPolar);

dataHandlers.registerNewType('customtype', function (html, parameters) {
  $(html).append(JSON.stringify(parameters) + '<br />');
  return {
    newData: function (data) {
      $(html).append(JSON.stringify(data) + '<br />');
    }
  }
});
