
class Log:
    def __init__(self, dir):
        self.dir = dir
        #timestamp
        ts = ...
    
    def test(self, val):
        filename = ...
        with open(filename, 'a') as f:
            f.write(val)

    def train(self, val):
        pass

    def lfw(self, val):
        pass

    def loss(self, val):
        pass
