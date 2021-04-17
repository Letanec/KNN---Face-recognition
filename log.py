import time

class Log:
    def __init__(self, filename):
        self.filename = filename + '_' + str(calendar.timegm(time.gmtime()))
        open(self.filename, "w").close()

    def print(self, iteration, epoch, data):
        printsel(self.filename, ": iteration", iteration, ", epoch", epoch, " -> ", data)
        with open(self.filename, "a") as f:
            f.write(str(iteration) + ',' + str(data) + '\n')