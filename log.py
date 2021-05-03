import time
import calendar
import os

class Log:
    def __init__(self, filename):
        self.filename = filename + '_' + str(calendar.timegm(time.gmtime()))
        open(self.filename, "w").close()

    def print(self, iteration, epoch, data):
        print(self.filename, ": iteration", iteration, ", epoch", epoch, " -> ", data)
        with open(self.filename, "a") as f:
            f.write(str(iteration) + ',' + str(data) + '\n')

def create_logs():
    path = "outputs"
    logs_names = ["acc","test_acc","ver","lfw_ver","loss","tf","lfw_tf"]
    logs = []
    for ln in logs_names:
        log = Log(os.path.join(path, ln))
        logs.append(log)
    return logs