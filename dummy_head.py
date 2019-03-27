import serial
import time
import numpy as np
class DummyHead():


    def __init__(self, port):
        """Will work in deg in this class"""
        self.ino = serial.Serial('/dev/cu.usbmodem'+str(port), 115200, timeout=1)
        time.sleep(2)
        self.theta = 0

        #need big range here
        self.max_left = 30
        self.max_right = -30

        #Starting moving left
        self.left = True
        self.right = False

    def look_around(self):
        """Move the dummy hehad so it looks around randomly"""

        # go back and forth
        if self.left:
            self.theta += 1

            if self.theta >= self.max_left:
                self.left = False
                self.right = True

        elif self.right:
            self.theta -= 1

            if self.theta <= self.max_right: #past -90 deg
                self.left = True
                self.right = False

        # self.theta = 20
        self.move(self.theta)

    def get_rotation(self):
        return np.deg2rad(self.theta)

    def close(self):
        self.ino.close()

    def move(self, theta):
        """Moves dummy head to location"""

        q1 = theta
        q2 = 0

        msg = self.formatAngle(q1,q2)
        # print(msg)
        self.ino.write(msg)

    def formatAngle(self, q1,q2):
        q1, q2 = self.calibrate(q1, q2)
        data =  str(q1) + "," + str(q2) + '\n'
        data =  str(q1) + "," + str(q2) + '\n'
        return data.encode()

    def calibrate(self, q1_angle, q2_angle):
        #make this so
        q1_offset = 90 - 30
        q2_offset = 0

        # q1_angle *= -1
        return q1_angle + q1_offset, q2_angle + q2_offset
    def read(self):
        data = self.ino.readline()
        if data:
            data = data.decode('ascii')
            data = data.strip('\n')
            data = data.strip('\r')
            print(data)




# #Example Usage
# head = DummyHead(14141)
# angle = 0
# while True:
#     time.sleep(0.5)
#     head.move(90)
#     # head.move(90)
#     # angle += np.deg2rad(1)#hhead has 1 deg of res
#     head.read()
# head.close()
