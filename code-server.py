#!/usr/local/bin/python
from wsgiref.simple_server import make_server
from ws4py.websocket import WebSocket
from ws4py.server.wsgirefserver import WSGIServer, WebSocketWSGIRequestHandler
from ws4py.server.wsgiutils import WebSocketWSGIApplication
from multiprocessing import Process, Queue
import subprocess
from os import chmod
import time
import sys
import socket
import json

def runCode(message, client, q):
    data = message.data
    print client
    print "start"
    with open('base-program.py', 'r') as baseFile:
        baseCode = baseFile.read()
    baseCode = baseCode.replace('#####INSERT: Here insert code\n', data)
    baseFile.close()
    filename = "code-program.py"
    file = open(filename, "w")
    file.write(baseCode)
    file.close()
    chmod(filename, 0700)
    # in fact this program will create a websocket server
    # we need to pass the port to the browser
    popen = subprocess.Popen(["python", "-u", filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, bufsize=1)
    try:
        client.send('{"port": 9001}', False)
    except socket.error, e:
        return
    except IOError, e:
        return
    q.put(popen)
    errorThread = Process(target = sendErr, args = (popen, client))
    errorThread.start()
    for l in execute(popen):
        try:
            client.send(json.dumps({'line': l}), False)
        except socket.error, e:
            break
        except AttributeError, e:
            break
        except IOError, e:
            break

    print "close"
    if (popen.poll() == None):
        popen.kill()
        print "killed"
    return_code = popen.wait()
    client.send(json.dumps({'status': 'ended', 'code': return_code}))

def sendErr(popen, client):
    stderr_lines = iter(popen.stderr.readline, "")
    for stderr_line in stderr_lines:
        client.send(json.dumps({'error': stderr_line}), False)
    print "laaa - error"
    popen.stderr.close()
    print "fini - error"

def execute(popen):
    stdout_lines = iter(popen.stdout.readline, "")
    for stdout_line in stdout_lines:
        yield stdout_line
    print "laaa"
    popen.stdout.close()
    return_code = popen.wait()
    # print return_code
    print "fini"

class EchoWebSocketMaison(WebSocket):
    def opened(self):
        print "New client"

    def received_message(self, message):
        print self
        print "New message"
        if message.data != "STOP":
            self.q_popen = Queue()
            self.scriptThread = Process(target = runCode, args = (message, self, self.q_popen))
            self.scriptThread.start()
        else:
            print "Kill Process"
            self.q_popen.get().kill()
            # print self.scriptThread.terminate()
            # print self.__popen.kill()

server = make_server('', 9000, server_class=WSGIServer,
                     handler_class=WebSocketWSGIRequestHandler,
                     app=WebSocketWSGIApplication(handler_cls=EchoWebSocketMaison))
server.initialize_websockets_manager()
server.serve_forever()
