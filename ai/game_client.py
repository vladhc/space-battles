import json
import socket

class RPSClient():
    def new_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', 6000))
        return s.makefile('rw')
    def __init__(self):
        self.io = self.new_socket()
    def reset(self):
        self.io.close()
        self.io = new_socket()

    def action(self, data):
        self.io.write('%s\n' % (data,))
        self.io.flush()
    def game_state(self):
        while 1:
            data = io.readline().strip()
            if not data:
                continue
            elif data[0] == "{":
                return json.loads(data)
