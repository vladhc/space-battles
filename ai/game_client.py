import string
import random
import json
import socket


class RPSClient():

    def __init__(self, username=None, password=None):
        if not username:
            username = randomString()
        if not password:
            password = randomString()
        self.username = username
        self.password = password
        self.io = self.new_socket()
        self.login()

    def new_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', 6000))
        return s.makefile('rw')

    def login(self):
        self.action('login {} {}'.format(self.username, self.password))

    def reset(self):
        self.io.close()
        self.io = self.new_socket()
        self.login()

    def action(self, data):
        self.io.write("{}\n".format(data))
        self.io.flush()

    def game_state(self):
        while 1:
            data = self.io.readline().strip()
            if not data:
                continue
            elif data[0] == "{":
                return json.loads(data)

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))
