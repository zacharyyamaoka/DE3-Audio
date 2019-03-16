import numpy as np
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "utils"))

from data_utils import get_zero_string



class LabelFile():

    def __init__(self,num,stem="data_rec_",path="/Users/zachyamaoka/Documents/de3_audio/data_real_label/"):

      self.PATH = path
      self.file_stem = stem
      num = get_zero_string(num)
      file_path =   self.PATH + self.file_stem + num + '.txt'
      self.file = open(file_path,"w+")


    def write_pos(self,x,y,z):
        prec = 3
        x = np.round(x,prec)
        y = np.round(y,prec)
        z = np.round(z,prec)

        line = str(x) + " " + str(y) + " " + str(z)
        self.file.write(line + "\n")
    def write_heading(self,theta):
        prec = 3
        theta = np.round(theta,prec)

        line = str(theta)
        self.file.write(line + "\n")

    def close(self):
        self.file.close()


#
# file = LabelFile(10)
# for i in range(10):
#     file.write_pos(0.12,0,0)
