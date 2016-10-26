#!/usr/local/bin/python
from wsgiref.simple_server import make_server
from ws4py.websocket import WebSocket
from ws4py.server.wsgirefserver import WSGIServer, WebSocketWSGIRequestHandler
from ws4py.server.wsgiutils import WebSocketWSGIApplication
from multiprocessing import Process
import subprocess
from os import chmod
import time
import sys
import socket

def runCode(message, client):
    data = message.data
    print "start"
    filename = "code-program.py"
    file = open(filename, "w")
    file.write("#!/usr/local/bin/python\n" + data)
    file.close()
    chmod(filename, 0700)
    # in fact this program will create a websocket server
    # we need to pass the port to the browser
    try:
        client.send('{"port": 9001}', False)
    except socket.error, e:
        return
    except IOError, e:
        return
    popen = subprocess.Popen(["python", "-u", filename], stdout=subprocess.PIPE, universal_newlines=True, bufsize=1)
    for l in execute(popen):
        try:
            client.send(l, False)
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

def execute(popen):
    stdout_lines = iter(popen.stdout.readline, "")
    for stdout_line in stdout_lines:
        yield stdout_line
    print "laaa"
    popen.stdout.close()
    return_code = popen.wait()
    print "fini"

class EchoWebSocketMaison(WebSocket):
    def opened(self):
        print "New client"

    def received_message(self, message):
        print "New message"
        if message.data != "STOP":
            self.scriptThread = Process(target = runCode, args = (message, self))
            self.scriptThread.start()
        else:
            print "Kill Process"
            self.scriptThread.terminate()


server = make_server('', 9000, server_class=WSGIServer,
                     handler_class=WebSocketWSGIRequestHandler,
                     app=WebSocketWSGIApplication(handler_cls=EchoWebSocketMaison))
server.initialize_websockets_manager()
server.serve_forever()
