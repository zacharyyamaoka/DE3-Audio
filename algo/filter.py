import numpy as np

class PositionFilter():

    def __init__(self):
        print("PositionFilter Init")
        self.pointer = 0
        #params for Moving Average Filter
        self.size = 20
        self.theta_mu = np.zeros(self.size)
        self.theta_var = np.zeros(self.size)

    def filter(self, last_theta_mu, last_theta_var):

        # simple moving average filter.

        self.theta_mu[self.pointer] = last_theta_mu
        self.theta_var[self.pointer] = last_theta_var
        #print(self.r_a)
        #print(self.theta_a)
        self.pointer += 1
        self.pointer = self.pointer % self.size #add wrap around

        curr_theta_mu = np.mean(self.theta_mu)
        curr_theta_var = np.var(self.theta_var)
        curr_theta_var = np.mean(self.theta_var)
        
        return curr_theta_mu, curr_theta_var
