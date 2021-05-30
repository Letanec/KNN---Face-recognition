from datetime import datetime


class Log:
    def __init__(self, dir):
        self.dir = dir
        self.ts = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        
    def test(self, val):
        filename = self.dir + "/test_" + self.ts + ".out"
        with open(filename, 'a') as f:
            f.write(str(val) + "\n")

    def train(self, val):
        filename = self.dir + "/train_" + self.ts + ".out"
        with open(filename, 'a') as f:
            f.write(str(val) + "\n")

    def loss(self, val):
        filename = self.dir + "/loss_" + self.ts + ".out"
        with open(filename, 'a') as f:
            f.write(str(val) + "\n")
