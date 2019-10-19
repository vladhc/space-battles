import json
import socket

class RPSClient():
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.io = self.new_socket()
        self.login()

    def new_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', 6000))
        return s.makefile('rw')

    def login(self):
        self.action('login %s %s' % (self.username, self.password))

    def reset(self):
        self.io.close()
        self.io = self.new_socket()
        self.login()

    def action(self, data):
        self.io.write('%s\n' % (data,))
        self.io.flush()

    def game_state(self):
        while 1:
            data = self.io.readline().strip()
            if not data:
                continue
            elif data[0] == "{":
                return json.loads(data)
