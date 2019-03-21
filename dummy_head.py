import serial
import time
import numpy as np
class DummyHead():


    def __init__(self, port):
        pass
        self.ino = serial.Serial('/dev/cu.usbmodem'+str(port), 115200, timeout=1)
        time.sleep(2)

    def move(self, theta):
        """Moves dummy head to location"""

        q1 = np.rad2deg(theta)
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
        q1_offset = 0
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

# head = DummyHead(14141)



#Example Usage

# angle = 0
# while True:
#     time.sleep(0.5)
#     head.move(angle)
#     angle += np.deg2rad(1)#hhead has 1 deg of res
#     head.read()
