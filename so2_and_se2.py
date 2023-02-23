import numpy as np

class   SO2:
    def  __init__(self, theta):
        self.theta = theta

    def hat(a):
        return np.array([[0, -a], [a, 0]])


class   SE2:
    def __init__(self, theta, x, y):
        self.theta = theta
        self.x = x
        self.y = y