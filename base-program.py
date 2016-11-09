#!/usr/local/bin/python

from ws4py.client.threadedclient import WebSocketClient
from wsgiref.simple_server import make_server
from ws4py.websocket import WebSocket
from ws4py.server.wsgirefserver import WSGIServer, WebSocketWSGIRequestHandler
from ws4py.server.wsgiutils import WebSocketWSGIApplication
from threading import Thread
from Queue import Queue
import os
import json
import sys
import socket
import json

r_messages = Queue()

# name: name of the graph
# id: choose a uniq alphanumeric identifier for this chart
# rendrer: type of the graph: 'area', 'line', 'bar', 'scatterplot'
# xName: name of x composant
# yName: name of y composant
# seriesNames: names of the series, orders: ['serie 1', 'serie 2', 'serie 3']
def createChart(name, id, renderer, xName, yName, seriesNames):
    r_messages.put(json.dumps({'newChart': name, 'id': id, 'renderer': renderer, 'xName': xName, 'yName': yName, 'seriesNames': seriesNames}))

rate = -1
channels = -1
buffer_frames = -1
volume = -1

#####INSERT: Here insert code

class StreamClient(WebSocketClient):
    # def opened(self):
    #     def data_provider():
    #         for i in range(1, 200, 25):
    #             yield "#" * i
    #
    #     self.send(data_provider())
    #
    #     for i in range(0, 200, 25):
    #         print i
    #         self.send("*" * i)
    #
    # def closed(self, code, reason=None):
    #     print "Closed down", code, reason

    def received_message(self, m):
        if not m.is_binary:
            print m
            global rate
            global channels
            global buffer_frames
            global volume
            m = json.loads(m.data)
            rate = m['rate']
            channels = m['channels']
            buffer_frames = m['buffer_frames']
            volume = m['volume']
        else:
            data = bytearray()
            data.extend(m.data)
            ndata = []
            i = 0
            for i in range(len(data) / 2):
                if data[2*i+1] <= 127:
                    ndata.append(data[2*i] + 256*data[2*i+1])
                else:
                    ndata.append((data[2*i+1]-128)*256 + data[2*i] - 32768)
            handleData(ndata)

def sendAudio(buffer):
    if client != -1:
        try:
            nbuffer = []
            for i in range(len(buffer)):
                if buffer[i] >= 0:
                    nbuffer.append(min(buffer[i]%256, 255))
                    nbuffer.append(min(buffer[i]/256, 255))
                else:
                    t = max(0, 32768 + buffer[i])
                    nbuffer.append(min(t%256, 255))
                    nbuffer.append(min(t/256 + 128, 255))
            client.send(bytearray(nbuffer), True)
            # print "send11"
        except socket.error, e:
            print "autre erreur11"
        except IOError, e:
            print "exception11"

client = -1

class WSServer(WebSocket):
    def opened(self):
        global client
        client = self
        while True:
            self.send(r_messages.get(), False)
            r_messages.task_done()

    def close(self, code, reason):
        global client
        global server
        global ws
        sys.stderr.write("1.1\n")
        client = -1
        os._exit(1)
        # print "ok"
        # # server.server_close()
        # print "la"
        # thread.interrupt_main()
        # print "cool"

def startServer(port):
    global server
    server = make_server('', port, server_class=WSGIServer,
                         handler_class=WebSocketWSGIRequestHandler,
                         app=WebSocketWSGIApplication(handler_cls=WSServer))
    server.initialize_websockets_manager()
    server.serve_forever()

try:
    global ws
    serverThread = Thread(target = startServer, args = (9001, ))
    serverThread.start()

    ws = StreamClient('ws://192.168.7.2:8081/', protocols=['http-only', 'chat'])
    ws.connect()

    ws.run_forever()
    serverThread.join()
except KeyboardInterrupt:
    sys.stderr.write("1.2\n")
    # ws.close()
