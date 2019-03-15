import numpy as np

class PositionFilter():

    def __init__(self):
        print("PositionFilter Init")
        self.pointer = 0


        #params for Moving Average Filter
        self.size = 20
        self.r_a = np.zeros(self.size)
        self.theta_a = np.zeros(self.size)
    def filter(self, last_r, last_theta):

        # simple moving average filter.

        self.r_a[self.pointer] = last_r
        self.theta_a[self.pointer] = last_theta
        #print(self.r_a)
        #print(self.theta_a)
        self.pointer += 1
        self.pointer = self.pointer % self.size #add wrap around

        r = np.mean(self.r_a)
        theta = np.mean(self.theta_a)

        return r, theta
