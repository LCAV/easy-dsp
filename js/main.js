var boardIp = location.host.replace(/:.+$/, '');
// var boardIp = '192.168.7.2';
var pythonDaemon = '127.0.0.1';

// ACE Code Editor
var aceEditor = ace.edit("editor");
var aceRange = ace.require('ace/range').Range
var AcePythonMode = ace.require("ace/mode/python").Mode;
aceEditor.session.setMode(new AcePythonMode());

// WebSockets
var wsAudio, wsConfig;
var wsPythonServer = new WebSocket("ws://" + pythonDaemon + ":7320");

var inputStream;   // Stream from WSAudio
var outputStream;  // Stream from python code
var outputHandle;
var b;
var audioCt = new AudioContext;
var config;

// Editor
var btnCodeStart = $('#btn-code-start');
var btnCodeStop = $('#btn-code-stop');
var cardEditor = $('#card-editor');

// Console
var cardConsole = $('#card-console');
var outputConsole = $('#console');
var outputReturnCode = $('#output-return-code');
var animationRunning = $('#animation-running');
animationRunning.hide();

// Message about external script
var cardExternalScript = $('#card-external-script');
var btnExternalScriptStop = $('#btn-external-script-stop');
cardExternalScript.hide();

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

wsPythonServer.onopen = function() {
  changeBadgeStatus(infosStatusPythonServer, 'connected');
  btnCodeStart.removeAttr('disabled');
  wsPythonServer.send(JSON.stringify({board: boardIp}));
};
wsPythonServer.onclose = function() {
  changeBadgeStatus(infosStatusPythonServer, 'disconnected');
  btnCodeStart.attr('disabled', 'disabled');
};

// Connection to daemons
function daemonsConnect() {
  wsAudio = new WebSocket("ws://" + boardIp + ":7321");
  wsConfig = new WebSocket("ws://" + boardIp + ":7322");
  wsAudio.onopen = function(e) {
    addAlert('success', 'Connected', 'Now connected to the audio stream');
    changeBadgeStatus(infosStatusWSAudio, 'connected');
  };
  wsConfig.onopen = function(e) {
    addAlert('success', 'Connected', 'Now connected to the control stream');
    changeBadgeStatus(infosStatusWSConfig, 'connected');
    configEnable();
  };
  wsAudio.onerror = function(e) {
    addAlert('danger', 'Error', 'Impossible to connect to the audio websocket');
  };
  wsConfig.onerror = function(e) {
    addAlert('danger', 'Error', 'Impossible to connect to the control websocket');
    configDisable();
  };
  wsAudio.onclose = function(e) {
    addAlert('danger', 'Error', 'Disconnected from the audio websocket');
    if (inputStream) {
      inputStream.destroyAudio();
    }
    changeBadgeStatus(infosStatusWSAudio, 'disconnected');
  };
  wsConfig.onclose = function(e) {
    addAlert('danger', 'Error', 'Disconnected from the control websocket');
    changeBadgeStatus(infosStatusWSConfig, 'disconnected');
    configDisable();
  };
  wsAudio.onmessage = onWSAudioMessage;
}
// Initially we try to connect
daemonsConnect();

// Restarting/Stopping board daemons
var btnDaemonsRestart = $('#btn-daemons-restart');
var btnDaemonsStop = $('#btn-daemons-stop');
btnDaemonsRestart.click(function() {
  $.get('api.php?restart=1', function (result) {
    console.log("Daemons restarted");
    // We reconnect
    daemonsConnect();
  });
});
btnDaemonsStop.click(function() {
  $.get('api.php?stop=1', function (result) {
    console.log("Daemons stopped");
  });
});

// Auto save of the editor
function restoreEditor() {
  var codeSaved = localStorage.getItem('editor.value');
  var cursorRow = parseInt(localStorage.getItem('editor.cursor.row'));
  var cursorColumn = localStorage.getItem('editor.cursor.column');
  if (codeSaved) {
    aceEditor.setValue(codeSaved, -1);
    aceEditor.moveCursorTo(cursorRow, cursorColumn);
    setTimeout(function () {
      aceEditor.scrollToLine(cursorRow+1, true, false);
    }, 500);
  }
}
restoreEditor();
function saveEditor() {
  var cursor = aceEditor.getCursorPosition();
  localStorage.setItem('editor.value', aceEditor.getValue());
  localStorage.setItem('editor.cursor.row', cursor.row);
  localStorage.setItem('editor.cursor.column', cursor.column);
  localStorage.setItem('editor.date', new Date());
}
aceEditor.on("change", _.throttle(saveEditor, 5000));

// Audio recording
var btnRecording = $('#btn-recording');
var textRecording = $('#btn-recording .text');
var timeRecording = $('#btn-recording .time');
var inRecording = false;
var recording;
var recordingTimer;
var recordingStart;
btnRecording.click(function() {
  if (!inRecording) {
    recording = new Recorder(inputStream.source, {workerPath: 'vendors/recorder.js', numChannels: config.channels});
    recording.record();
    inRecording = true;
    btnRecording.removeClass('btn-danger');
    btnRecording.addClass('btn-secondary');
    textRecording.text('Stop');
    recordingStart = new Date();
    recordingTimerUpdate();
  } else {
    clearTimeout(recordingTimer);
    recording.stop();
    recording.exportWAV(function (file) {
      var a = document.createElement("a");
      document.body.appendChild(a);
      a.style = "display: none";
      a.href = window.URL.createObjectURL(file);
      a.download = 'recording.wav';
      a.click();
    });
    btnRecording.removeClass('btn-secondary');
    btnRecording.addClass('btn-danger');
    textRecording.text('Record');
    inRecording = false;
  }
});
function recordingTimerUpdate() {
  recordingTimer = setTimeout(recordingTimerUpdate, 30 + Math.floor(20*Math.random()));
  var diff = ((new Date()) - recordingStart);
  var display = zeros(Math.floor(diff/1000/60)) + ':' + zeros(Math.floor((diff/1000)%60)) + '.' + zeros((diff%1000), 2);
  timeRecording.text(display);
}
function zeros(n, number) {
  var out = n;
  if (n < 10) {
    out = '0' + out;
  }
  if (number == 2 && n < 100) {
    out = '0' + out;
  }
  return out;
}

// External script
function startExternalScript() {
  cardEditor.slideUp();
  cardConsole.slideUp();
  cardExternalScript.slideDown();
}
function endExternalScript() {
  cardEditor.slideDown();
  cardConsole.slideDown();
  cardExternalScript.slideUp();
}
btnExternalScriptStop.click(function() {
  outputHandle.close();
  endExternalScript();
});

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
var outputBuffered = '';
var printOutputBufferedTimeout;

// We must buffer the output, and add it to the HTML every 100ms
// Else the interface will freeze
// This function will be called every 100ms
function printOutputBuffered() {
  // Should we maintain a scroll to the end of the output console?
  var end = outputConsole.scrollTop() == (outputConsole[0].scrollHeight - outputConsole.height());
  outputConsole.html(outputConsole.html() + outputBuffered);
  if (end) {
    outputConsole.scrollTop((outputConsole[0].scrollHeight - outputConsole.height()));
  }

  outputBuffered = '';
  printOutputBufferedTimeout = setTimeout(printOutputBuffered, 100);
}

// Start
var codeRunning = false;
function executeCode() {
  if (codeRunning) {
    return;
  }
  codeRunning = true;
  btnCodeStart.attr('disabled', 'disabled');
  animationRunning.show();

  // We clean the console
  outputConsole.html('');
  outputReturnCode.html('');

  // Remove errors markers
  _.forEach(aceEditor.session.getMarkers(), function(marker) {
    if (marker.clazz == 'bg-error') {
      aceEditor.session.removeMarker(marker.id);
    }
  });

  // Every 100ms we will display the ouput buffered
  printOutputBufferedTimeout = setTimeout(printOutputBuffered, 100);

  wsPythonServer.send(aceEditor.getValue());
  btnCodeStop.removeAttr('disabled');
}
btnCodeStart.click(executeCode);
// Stop
function stopCode() {
  if (!codeRunning) {
    return;
  }
  codeRunning = false;
  wsPythonServer.send("STOP");
  clearTimeout(printOutputBufferedTimeout);
  outputHandle.close();
}
btnCodeStop.click(stopCode);

// Key shortcuts
function toggleCodeLaunch() {
  if (codeRunning) {
    stopCode();
  } else {
    executeCode();
  }
}
key('⌘+e', toggleCodeLaunch); // outside the editor
aceEditor.keyBinding.addKeyboardHandler(function (editor, cmd, char, touch) { // in the editor
  if (cmd == 8 && touch == 69) { // ⌘+e
    toggleCodeLaunch();
  }
});

var codeFirstLine;
var scriptLaunchedPort = -1;
wsPythonServer.onmessage = function(e) {
  var message = JSON.parse(e.data);
  if (message.port) { // A script has be launched on port message.port
    scriptLaunchedPort = message.port;
  } else if (message.line || message.error) { // Simple output from stdout or stderr from the programm
    var line = message.line;
    if (message.error) {
      message.error = message.error.replace(/File "code-program\.py"\, line ([0-9]+)/g, function (l, n) {
        n = parseInt(n);
        var nbOfLines = aceEditor.getValue().match(/[\n]/g).length + 1;
        if (n >= codeFirstLine && n <= (codeFirstLine + nbOfLines)) {
          aceEditor.session.addMarker(new aceRange(n-codeFirstLine, 0, n-codeFirstLine, 5), "bg-error", "fullLine");
          return 'File "code-program.py", line ' + (n-codeFirstLine);
        } else {
          return l;
        }
      });
      line = '<span class="line-error">' + message.error + '</span>';
    }

    outputBuffered += line;
  } else if (message.codeLine) {
    codeFirstLine = message.codeLine;
  } else if (message.status) { // Status information
    if (message.status == 'ended') {
      btnCodeStart.removeAttr('disabled');
      btnCodeStop.attr('disabled', 'disabled');
      outputReturnCode.html(' - Code: ' + message.code);
      animationRunning.hide();
      codeRunning = false;
    }
  } else if (message.script) {
    if (scriptLaunchedPort != message.script) {
      startExternalScript();
    }
    scriptLaunchedPort = -1;
    setTimeout(function() {
      outputHandle = new handleOutput(message.script);
    }, 300);
  } else {
    console.warn("wsPythonServer.onmessage(): Unknown message", message);
  }
};

// Input stream: audio and configuration
function onWSAudioMessage(e) {
  if (typeof e.data == "string") { // configuration


    var conf = JSON.parse(e.data);

    if(conf.hasOwnProperty('rate')){ // new config

      config = conf;
      console.log("Config received:", config);
      displayConfig();

      // If the audio was playing, we need to recreate
      // the stream using the new configuration

      if (inputStream) {
        // check the current status of the source
        var was_playing = !(inputStream.isStopped());

        // destroy it
        inputStream.destroyAudio();

        // if it was playing before, restart it
        if (was_playing) {
          inputStream = new sourceAudio(audioCt, conf);
        }
      }
    } else if( conf.hasOwnProperty('possible_channel') ) { // supported config info

        var channels = conf.possible_channel;
        var sel = document.getElementById('config-channels');
        for(var i = 0; i < channels.length; i++) {
            var opt = document.createElement('option');
            opt.innerHTML = channels[i];
            opt.value = channels[i];
            sel.appendChild(opt);
        }

        var rates = conf.possible_rates;
        var sel = document.getElementById('config-rate');
        for(var i = 0; i < rates.length; i++) {
            var opt = document.createElement('option');
            opt.innerHTML = rates[i];
            opt.value = rates[i];
            sel.appendChild(opt);
        }

    }

    return;


  }
  b = e;
  // console.log(e.data);
  if (inputStream) {
    inputStream.loadData(e.data);
  }
}

var inputConfigRate = $('#config-rate');
var inputConfigBuffer = $('#config-buffer');
var inputConfigChannels = $('#config-channels');
var inputConfigVolume = $('#config-volume');
var btnConfigChange = $('#config-change');
var configElements = [inputConfigRate, inputConfigBuffer, inputConfigChannels, inputConfigVolume, btnConfigChange];
btnConfigChange.click(function() {
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

function configDisable() {
  _.forEach(configElements, function (e) {
    e.attr('disabled', 'disabled');
  });
}
function configEnable() {
  _.forEach(configElements, function (e) {
    e.removeAttr('disabled');
  });
}

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

// Audio latency
var infosAudioLatency = $('#info-audio-latency');

// Manage the connection with the running code
function handleOutput(port) {
  var ws = new WebSocket("ws://" + pythonDaemon + ":" + port);
  var nbTry = 1;
  var stop = false; // se to true when the close of this handleOutput is asked

  // Try multiple times to connect
  // Useful because we don't know after how much time

  // Remove previous tabs
  $('#output-tabs .output-tab').remove();
  $('#output-panes .output-pane').remove();

  ws.onmessage = function(e) {
    if (typeof e.data != "string") { // we have binary data: it's audio
      if (!outputStream || outputStream.isStopped()) {
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
        $('#output-tab-' + data.id).tab('show');
        dataHandlers.addHandler(data.id, data.type, document.getElementById('graph-' + data.id), data.parameters);
      } else if (data.dataHandler) {
        dataHandlers.addNewData(data.dataHandler, data.data);
      } else if (data.latency) {
        infosAudioLatency.text(Math.ceil(data.latency) + ' ms');
        if (data.latency < 400) {
          infosAudioLatency.removeClass('tag-danger');
          infosAudioLatency.removeClass('tag-warning');
          infosAudioLatency.addClass('tag-success');
        } else if (data.latency < 1000) {
          infosAudioLatency.removeClass('tag-danger');
          infosAudioLatency.removeClass('tag-success');
          infosAudioLatency.addClass('tag-warning');
        } else {
          infosAudioLatency.removeClass('tag-warning');
          infosAudioLatency.removeClass('tag-success');
          infosAudioLatency.addClass('tag-danger');
        }
      } else {
        console.warn("handleOutput - ws.onmessage: unknow message type:", data);
      }
    }
  };

  ws.onclose = function (e) {
    if (outputStream) {
      outputStream.destroyAudio();
    }
  }

  function stopAudio() {
    if (outputStream) {
      outputStream.destroyAudio();
    }
  }
  function playAudio() {
    outputStream = new sourceAudio(audioCt, config);
  }
  function close() {
    stop = true;
    stopAudio();
    ws.close();
  }

  return {
    stopAudio: stopAudio,
    playAudio: playAudio,
    close: close
  };
}

// This function receives audio from the network convert the samplerate to the
// local one and sends it to play. Note that the simple linear interpolation
// used here will not work well if the two sampling rates are wildly different
// 
// This function should also manage the play/stop buttons
//
function sourceAudio(audioCtx, config) {

  var channelData, buffer_size;

  // state of the source
  var playing = false;
  var stopped = false;

  // We'll cache a few buffer to avoid glitches
  var cache = [];
  var cache_min_length = 5;  // minimum buffering time

  // This is the running buffer
  buffer_size = 256;
  var current_buffer = audioCtx.createBuffer(config.channels, buffer_size, config.rate);
  var current_buffer_filling = 0;

  // channels of the running buffer
  channelData = [];
  for (var i = 0; i < config.channels; i++) {
    channelData[i] = current_buffer.getChannelData(i);
  }

  // we need to implement resampling from source rate to audio context rate
  var time_step = config.rate / audioCtx.sampleRate;
  var frac_time = 0.;
  var state = new Array(config.channels).fill(0.);
  
  // This is a node responsible for audio processing
  var scriptNode = audioCtx.createScriptProcessor(buffer_size, config.channels, config.channels);
  
  // Give the node a function to process audio events
  scriptNode.onaudioprocess = function(audioProcessingEvent) {

    // The output buffer contains the samples that will be modified and played
    var outputBuffer = audioProcessingEvent.outputBuffer;

    if (playing == true || (playing == false && cache.length > cache_min_length))
    {
      playing = true;
      var inputBuffer = cache.shift();

      // Loop through the output channels (in this case there is only one)
      for (var channel = 0; channel < outputBuffer.numberOfChannels; channel++) {
        if (inputBuffer) {
          var inputData = inputBuffer.getChannelData(channel);
          var outputData = outputBuffer.getChannelData(channel);

          // Loop through all samples in the buffer
          for (var sample = 0; sample < inputBuffer.length; sample++) {
            // make output equal to the same as the input
            outputData[sample] = inputData[sample];
          }
        }
      }
    } else {
      // Just fill with zeros until we get some audio
      for (var channel = 0; channel < outputBuffer.numberOfChannels; channel++) {
        var outputData = outputBuffer.getChannelData(channel);

        // Loop through all samples
        for (var sample = 0; sample < outputBuffer.length; sample++) {
          // make output equal to the same as the input
          outputData[sample] = 0;
        }
      }
    }

  }

  // Connect the script node to outside world
  scriptNode.connect(audioCtx.destination);

  // This function is responsible for filling the buffer cache with
  // data from the stream
  function loadData(data) {

    // if the source is stopped, do nothing
    if (stopped) {
      return;
    }

    // Otherwise, process incoming data
    var fileReader = new FileReader();

    fileReader.onload = function() {
      var data = new Int16Array(this.result);

      var n_samples = data.length / config.channels;

      while (frac_time <= n_samples - 1)
      {
        var n1 = Math.floor(frac_time);
        var n2 = Math.ceil(frac_time);

        for (var ch = 0 ; ch < config.channels ; ch++) {
          if (n1 < 0) {
            p1 = state[ch] / 256 / 128;
          } else {
            p1 = data[n1 * config.channels + ch] / 256 / 128;
          }
          p2 = data[n2 * config.channels + ch] / 256 / 128;

          // Linear interpolation
          channelData[ch][current_buffer_filling] = (p2 - p1) * (frac_time - n1) + p1;
        }

        // increment current buffer counter
        current_buffer_filling += 1;
        
        // Current buffer is full. We need a new buffer
        if (current_buffer_filling == buffer_size) {
          // add current buffer to the cache
          cache.push(current_buffer);

          // create a new buffer
          current_buffer = audioCtx.createBuffer(config.channels, buffer_size, config.rate);
          current_buffer_filling = 0;
          for (var ch = 0; ch < config.channels; ch++) {
            channelData[ch] = current_buffer.getChannelData(ch);
          }
        }

        // move time
        frac_time += time_step;
      }

      // Save the last sample of each channel in the state
      for (var ch = 0 ; ch < config.channels ; ch++) {
        state[ch] = data[data.length - config.channels + ch];
      }

      // Remove the data size from the fractional time
      frac_time -= n_samples;

    };

    fileReader.readAsArrayBuffer(data);
  }

  function destroyAudio() {
    if (stopped === false) {
      console.log("Stop audio source.");
      if (scriptNode.numberOfOutputs > 0) {
        scriptNode.disconnect(audioCtx.destination);
      }

      // empty cache and stop playing
      cache = [];
      playing = false;
      stopped = true;
    }
  }

  function isStopped() {
    return stopped;
  }

  return {
    loadData: loadData,
    destroyAudio: destroyAudio,
    source: scriptNode,
    isStopped: isStopped
  };
}

// Selection of the Python daemon
var btnPythonDmCustom = $('#python-dm-custom');
var displayPythonDmCustomIp = $('#python-dm-custom-ip');
btnPythonDmCustom.click(function () {
  var ip = prompt("Which IP address to use for the Python daemon?", pythonDaemon);
  pythonDaemon = ip;
  displayPythonDmCustomIp.text(ip);
  wsPythonServer = new WebSocket("ws://" + pythonDaemon + ":7320");
});
