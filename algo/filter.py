import numpy as np

class PositionFilter():
    # Implements 1D bayes filter for integrating data + enabling head tracking

    def __init__(self, res=10):
        print("PositionFilter Init")
        self.pointer = 0
        #params for Moving Average Filter
        self.size = 20
        self.theta_mu = np.zeros(self.size)
        self.theta_var = np.zeros(self.size)


        self.res = np.deg2rad(res)
        self.n = int(round(np.pi * 2 / self.res))
        self.step = np.pi * 2/float(self.n)



        # np.bel = np.zeros(self.n)
        # uniform init
        self.bel = np.ones(self.n) / float(self.n)

        self.drift = 1


    def filter(self, last_theta_mu, last_theta_var):

        # simple moving average filter.
        last_theta_mu = last_theta_mu % (2 * np.pi) #modulo
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

    def angle_delta(self, x, y):
        d = np.abs(x - y)
        d = np.abs(d - np.pi)
        d = np.pi - d

        return d

    def eval_gaussian(self, x, theta_mu, var):


        norm = 1 / np.sqrt(2 * np.pi * var)

        #find smallest angle
        d = self.angle_delta(x,theta_mu)
        power = - ((d) ** 2)/ (2*var)
        p = np.exp(power)/norm
        return p

    def bin_ind_2_theta(self, ind):
        return ((self.step * ind) + self.step/2)

    def motion_update(self,dt=0.1):
        #assume randomly left or right motion.....
        norm = 0
        new_bel = np.zeros(self.n)

        drift_constant = self.drift * dt #function of dt and drift rate

        for i in np.arange(self.n):
            new_p = 0
            theta = self.bin_ind_2_theta(i)
            for j in np.arange(self.n):
                theta_j = self.bin_ind_2_theta(j)
                d = self.angle_delta(theta,theta_j)

                mul = np.exp(-d/(2*drift_constant))
                new_p += mul * self.bel[j] #integrate belief from all thheta

            new_bel[i] = new_p
            norm += new_p
        new_bel /= norm
        self.bel = new_bel
        print("Motion Update")

    def eval_HRTF(self, x,theta_mu,var):
        # more binary
        d = self.angle_delta(x,theta_mu)
        if d < var/2:
            return 0.3
        else:
            return 0.1

    def sensor_update(self, theta_mu, var=np.pi): #update with sensor reading and accuracy

        new_bel = np.zeros(self.n)
        total_p = 0
        print("New Measuerment: ", theta_mu)
        for i in np.arange(self.n): #for each bin update with likelihood of measurement
            # x = (self.step * (i - 1)) + self.step/2 #find the center of the bin
            x = self.bin_ind_2_theta(i) #find the center of the bin

            likelihood = self.eval_gaussian(x,theta_mu,var)
            # likelihood = self.eval_HRTF(x,theta_mu,var)

            new_p = likelihood * self.bel[i]
            new_bel[i] = new_p
            total_p += new_p

        new_bel /= total_p #normalize afterwards

        self.bel = new_bel #replace old belief
        print("Sensor Update")


    def get_peak(self):
        #returns current best belief in location
        max = np.argmax(self.bel)
        theta = (self.step * max) + self.step/2 #inds start at 0
        return theta
