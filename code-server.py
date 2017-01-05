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
import signal

lineIdentifier = '#####INSERT: Here insert code';

def runCode(message, client, q):
    data = message.data
    print "runCode"
    with open('base-program.py', 'r') as baseFile:
        baseCode = baseFile.read()
    lines = baseCode.split('\n')
    lineNumber = 0
    for l in lines:
        lineNumber += 1
        if l == lineIdentifier:
            client.send(json.dumps({'codeLine': lineNumber}))
            break
    baseCode = baseCode.replace(lineIdentifier + '\n', data)
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
    popen.stderr.close()

def execute(popen):
    stdout_lines = iter(popen.stdout.readline, "")
    for stdout_line in stdout_lines:
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()

clients = []
board_ip = ''

class PythonDaemon(WebSocket):
    def opened(self):
        print "New client"
        clients.append(self)

    def received_message(self, message):
        global board_ip
        try:
            data = json.loads(message.data)
            print "New JSON"
            if 'script' in data:
                # Send a message to every connected client so it knows
                for c in clients:
                    if c != self:
                        c.send(json.dumps(data))
                # Send the board IP
                if board_ip != '':
                    self.send(board_ip)
            if 'board' in data:
                # The browser inform us of the IP address of the board
                board_ip = data['board']
            return
        except ValueError, e:
            doNothing = True
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

    def closed(self, code, reason=None):
        clients.remove(self)
        if hasattr(self, 'q_popen') and not self.q_popen.empty():
            self.q_popen.get().kill()

server = make_server('', 7320, server_class=WSGIServer,
                     handler_class=WebSocketWSGIRequestHandler,
                     app=WebSocketWSGIApplication(handler_cls=PythonDaemon))

def quit_everything(signal, frame):
    global server
    print 'Exiting...'
    server.server_close()
signal.signal(signal.SIGINT, quit_everything)

server.initialize_websockets_manager()
print "Listening..."
server.serve_forever()
