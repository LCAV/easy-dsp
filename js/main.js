// ACE Code Editor
var aceEditor = ace.edit("editor");
var aceRange = ace.require('ace/range').Range
var AcePythonMode = ace.require("ace/mode/python").Mode;
aceEditor.session.setMode(new AcePythonMode());

// WebSockets
var wsAudio = new WebSocket("ws://192.168.7.2:8081");
var wsConfig = new WebSocket("ws://192.168.7.2:8082");
var wsPythonServer = new WebSocket("ws://127.0.0.1:9000");

var inputStream;
var outputHandle;
var b;
var audioCt = new AudioContext;
var config;

// Editor
var btnCodeStart = $('#btn-code-start');
var btnCodeStop = $('#btn-code-stop');

// Console
var outputConsole = $('#console');
var outputReturnCode = $('#output-return-code');
var animationRunning = $('#animation-running');
animationRunning.hide();


// Alerts
var infosStatusPythonServer = $('#info-status-code');
var infosStatusWSAudio = $('#info-status-audio');
var infosStatusWSConfig = $('#info-status-control');
var alerts = $('#alerts');
function addAlert(type, title, message) {
  alerts.prepend('<div class="alert alert-' + type + '" role="alert"><strong>' + title + '</strong> ' + message + '</div>');
}
function changeBadgeStatus(e, status) {
  if (status == "connected") {
    e.text('Connected');
    e.addClass('tag-success');
    e.removeClass('tag-danger');
  } else if (status == "disconnected") {
    e.text('Disconnected');
    e.addClass('tag-danger');
    e.removeClass('tag-success');
  } else {
    console.warn("changeBadgeStatus(): Unknown badge status", e, status);
  }
}
wsAudio.onopen = function(e) {
  addAlert('success', 'Connected', 'Now connected to the audio stream');
  changeBadgeStatus(infosStatusWSAudio, 'connected');
};
wsConfig.onopen = function(e) {
  addAlert('success', 'Connected', 'Now connected to the control stream')
  changeBadgeStatus(infosStatusWSConfig, 'connected');
};
wsPythonServer.onopen = function() {
  changeBadgeStatus(infosStatusPythonServer, 'connected');
};
wsAudio.onerror = function(e) {
  addAlert('danger', 'Error', 'Impossible to connect to the audio websocket');
};
wsConfig.onerror = function(e) {
  addAlert('danger', 'Error', 'Impossible to connect to the control websocket');
};
wsAudio.onclose = function(e) {
  addAlert('danger', 'Error', 'Disconnected from the audio websocket');
  changeBadgeStatus(infosStatusWSAudio, 'disconnected');
};
wsConfig.onclose = function(e) {
  addAlert('danger', 'Error', 'Disconnected from the control websocket');
  changeBadgeStatus(infosStatusWSConfig, 'disconnected');
};
wsPythonServer.onclose = function() {
  changeBadgeStatus(infosStatusPythonServer, 'disconnected');
};

// Output Analyze
// Audio
var btnAudioInputPlay = $('#btn-audio-input-play');
var btnAudioInputStop = $('#btn-audio-input-stop');
var btnAudioOutputPlay = $('#btn-audio-output-play');
var btnAudioOutputStop = $('#btn-audio-output-stop');
btnAudioInputPlay.click(function() {
  if (inputStream) {
    inputStream.destroyAudio();
  }
  inputStream = new sourceAudio(audioCt, config);
});
btnAudioInputStop.click(function() {
  if (inputStream) {
    inputStream.destroyAudio();
  }
});
btnAudioOutputPlay.click(function() {
  outputHandle.playAudio();
});
btnAudioOutputStop.click(function() {
  outputHandle.stopAudio();
});

// Tabs
$('#output-tab-audio').click(function (e) {
  $(this).tab('show');
});


// Code execution
// Start
btnCodeStart.click(function() {
  btnCodeStart.attr('disabled', 'disabled');
  outputConsole.html('');
  animationRunning.show();
  outputReturnCode.html('');
  // Remove previous tabs
  $('#output-tabs .output-tab').remove();
  $('#output-panes .output-pane').remove();

  // Remove errors markers
  _.forEach(aceEditor.session.getMarkers(), function(marker) {
    if (marker.clazz == 'bg-error') {
      aceEditor.session.removeMarker(marker.id);
    }
  });
  wsPythonServer.send(aceEditor.getValue());
  btnCodeStop.removeAttr('disabled');
});
// Stop
btnCodeStop.click(function() {
  wsPythonServer.send("STOP");
  outputHandle.close();
});

wsPythonServer.onmessage = function(e) {
  var message = JSON.parse(e.data);
  // Information about which port to connect
  if (message.port) {
    setTimeout(function() {
      outputHandle = new handleOutput(message.port);
    }, 300);
  } else if (message.line || message.error) { // Simple output from stdout or stderr from the programm
    var line = message.line;
    if (message.error) {
      message.error = message.error.replace(/File "code-program\.py"\, line ([0-9]+)/g, function (l, n) {
        n = parseInt(n);
        var nbOfLines = aceEditor.getValue().match(/[\n]/g).length + 1;
        var firstLine = 31;
        if (n > firstLine && n <= (firstLine + nbOfLines)) {
          aceEditor.session.addMarker(new aceRange(n-firstLine-1, 0, n-firstLine-1, 5), "bg-error", "fullLine");
          return 'File "code-program.py", line ' + (n-firstLine);
        } else {
          return l;
        }
      });
      line = '<span class="line-error">' + message.error + '</span>';
    }
    // Should we maintain a scroll to the end of the output console?
    var end = outputConsole.scrollTop() == (outputConsole[0].scrollHeight - outputConsole.height());
    outputConsole.html(outputConsole.html() + line);
    if (end) {
      outputConsole.scrollTop((outputConsole[0].scrollHeight - outputConsole.height()));
    }
  } else if(message.status) { // Status information
    if (message.status == 'ended') {
      btnCodeStart.removeAttr('disabled');
      btnCodeStop.attr('disabled', 'disabled');
      outputReturnCode.html(' - Code: ' + message.code);
      animationRunning.hide();
    }
  } else {
    console.warn("wsPythonServer.onmessage(): Unknown message", message);
  }
};

// Input stream: audio and configuration
wsAudio.onmessage = function(e) {
  if (typeof e.data == "string") { // configuration
    var conf = JSON.parse(e.data);
    config = conf;
    console.log("Config received:", config);
    displayConfig();
    if (inputStream) {
      inputStream.destroyAudio();
    }
    inputStream = new sourceAudio(audioCt, conf);
    return;
  }
  b = e;
  // console.log(e.data);
  inputStream.loadData(e.data);
}
var inputConfigRate = $('#config-rate');
var inputConfigBuffer = $('#config-buffer');
var inputConfigChannels = $('#config-channels');
var inputConfigVolume = $('#config-volume');
$('#config-change').click(function() {
  var newConfig = {
    rate: parseInt(inputConfigRate.val()),
    buffer_frames: parseInt(inputConfigBuffer.val()),
    channels: parseInt(inputConfigChannels.val()),
    volume: parseInt(inputConfigVolume.val())
  };
  console.log("Send new config", newConfig);
  addAlert('info', 'Configuration sent', 'New configuration sent: ' + JSON.stringify(newConfig));
  wsConfig.send(JSON.stringify(newConfig));
});

function displayConfig() {
  inputConfigRate.val(config.rate);
  inputConfigChannels.val(config.channels);
  inputConfigBuffer.val(config.buffer_frames);
  inputConfigVolume.val(config.volume);
  addAlert('warning', 'Configuration change', 'New configuration received: ' + JSON.stringify(config));
  $('#info-rate').text(config.rate);
  $('#info-channels').text(config.channels);
  $('#info-frames').text(config.buffer_frames);
  $('#info-volume').text(config.volume);
}

// Manage the connection with the running code
function handleOutput(port) {
  var ws = new WebSocket("ws://127.0.0.1:" + port);
  var outputStream;

  ws.onmessage = function(e) {
    if (typeof e.data != "string") { // we have binary data: it's audio
      if (!outputStream) {
        outputStream = new sourceAudio(audioCt, config);
      }
      outputStream.loadData(e.data);
    } else {
      var data = JSON.parse(e.data);
      if (data.addHandler) {
        $('#output-tabs').append($('<li class="nav-item output-tab"><a class="nav-link" href="#output-pane-' + data.id + '" id="output-tab-' + data.id + '" role="tab">' + data.addHandler + '</a></li>'));
        $('#output-panes').append($('<div class="tab-pane output-pane" id="output-pane-' + data.id + '" role="tabpanel"><div class="graph" id="graph-' + data.id + '"></div></div>'));
        $('#output-tab-' + data.id).click(function (e) {
          e.preventDefault();
          $(this).tab('show');
        });
        dataHandlers.addHandler(data.id, data.type, document.getElementById('graph-' + data.id), data.parameters);
      } else if (data.dataHandler) {
        dataHandlers.addNewData(data.dataHandler, data.data);
      } else {
        console.warn("handleOutput - ws.onmessage: unknow message type:", data);
      }
    }
  };

  function stopAudio() {
    if (outputStream) {
      outputStream.destroyAudio();
    }
  }
  function playAudio() {
    outputStream = new sourceAudio(audioCt, config);
  }
  function close() {
    stopAudio();
    ws.close();
  }

  return {
    stopAudio: stopAudio,
    playAudio: playAudio,
    close: close
  };
}

// This function should also manage the play/stop buttons
function sourceAudio(audioCtx, config) {
  var audioData, channelData, audioPos, source, buffer_size;
  buffer_size = 2 * config.rate; // we want two seconds
  audioData = audioCtx.createBuffer(config.channels, buffer_size, config.rate); // channels - size of the buffer - frameRate
  channelData = [];
  for (var i = 0; i < config.channels; i++) {
    channelData[i] = audioData.getChannelData(i);
  }
  audioPos = 0;
  source = audioCtx.createBufferSource();
  source.loop = true;
  source.buffer = audioData;
  source.connect(audioCtx.destination);
  setTimeout(function () {
    console.log("start");
    source.start();
  }, 3000);

  function loadData(data) {
    var fileReader = new FileReader();
    fileReader.onload = function() {
      var data = new Int16Array(this.result);
      // console.log(data[0], data[100], data[1000], data[5000]);
      // console.log(data.length);
      // console.log(data);
      for (var i = 0; i < data.length; i++) {
        var channel = i % config.channels;
        channelData[channel][audioPos] = data[i]/256/128; // 16 bits audio => we must move it between -1 and 1
        if (channel == (config.channels - 1)) {
          audioPos = (audioPos + 1) % (buffer_size);
        }
      }
    };
    fileReader.readAsArrayBuffer(data);
  }

  function destroyAudio() {
    if (source !== undefined) {
      source.stop();
    }
  }

  return {
    loadData: loadData,
    destroyAudio: destroyAudio,
    source: source
  };
}
