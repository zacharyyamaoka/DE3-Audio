
import numpy as np
from pythonosc import osc_message_builder
from pythonosc import udp_client
import argparse

class TuneInControl():

    def __init__(self,port):
      self.POSITION = "/3DTI-OSC/source1/pos"
      self.PLAY = "/3DTI-OSC/source1/play"
      self.PAUSE = "/3DTI-OSC/source1/pause"

      parser = argparse.ArgumentParser()
      parser.add_argument("--ip", default="127.0.0.1",
          help="The ip of the OSC server")
      parser.add_argument("--port", type=int, default=port,
          help="The port the OSC server is listening on")
      args = parser.parse_args()
      self.client = udp_client.SimpleUDPClient(args.ip, args.port)

    def send_position(self,x,y,z):
        self.client.send_message(self.POSITION, (x,y,z))
    def play(self):
        self.client.send_message(self.PLAY,1)
    def pause(self):
        self.client.send_message(self.PAUSE,1)

class RandomPolarWalker():

    def __init__(self, rec_time = 1):

        self.r = np.random.uniform(0,4)
        self.theta = np.random.uniform(0,2*np.pi)
        self.z = 0

        # self.speed_mu = 1.4 # m/s - speed of slow walk https://en.wikipedia.org/wiki/Walking
        self.speed_var = 0
        self.speed_mu = 0
        self.acc_var = 0.86 #https://www.researchgate.net/post/What_is_the_maximum_walking_acceleration_deceleration_over_a_very_short_time_period_eg_002_01_05_sec
        self.acc_std = np.sqrt(self.acc_var)
        self.speed_std = np.sqrt(self.speed_var)

        self.r_dot = self.speed_mu
        self.theta_dot = self.speed_mu

        self.min_r = 0.20 # no closer then 20cm
        self.max_r = 4 # no furthher than 4m

        self.timer = 0
        self.slow_timer = 0
        self.accel = 0.5
        self.slow_speed = (2*np.pi)/ (rec_time)
        #
        if np.random.random() > 0.5:
            self.slow_speed *= -1

    def slow_update(self, dt=0.1): # this walker just goes around in circles at a more consitent and slow Rate
        self.slow_timer += dt

        self.r = 3# keep it constant....
        self.theta += self.slow_speed * dt
        self.theta = self.theta % (2 * np.pi)
        # print(np.rad2deg(self.theta))
    def update(self, dt=0.1):

        #update speed and orientation
        self.timer += dt

        self.r_dot += dt * np.random.normal(0,self.acc_std)
        self.theta_dot += dt * np.random.normal(0,self.acc_std) #in small steps....

        #Move person
        self.r += self.r_dot * dt
        self.theta += self.theta_dot * dt

        # with small probabality switch direction
        if self.timer > 3: # every one second you may switchh

            self.timer = 0

            #with small probability stop, mabye also fixes this unbounded increase problem
            if np.random.random() > 0.8:
                self.theta_dot = 0
            if np.random.random() > 0.8:
                self.r_dot = 0

        self.check_bounds()
        """Move the Random walker in space"""

    def heading(self):

        # map_theta = (self.theta + np.pi) % (2 * np.pi ) - np.pi
        # print(np.rad2deg(map_theta))
        return self.theta
    def location(self):

        x = self.r*np.cos(self.theta)
        y = self.r*np.sin(self.theta)
        z = self.z

        return x, y, z


    def check_bounds(self):
        """Make sure walker stays within room"""

        if self.r > self.max_r:
            self.r = self.max_r
            # self.r_dot = 0 #stop the person
            self.r_dot = -0.1 #send them back

        elif self.r < self.min_r:
            self.r = self.min_r
            # self.r_dot = 0
            self.r_dot = 0.1
