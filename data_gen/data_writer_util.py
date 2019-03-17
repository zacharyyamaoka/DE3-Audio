import numpy as np
import os
import sys
import wave
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

class BatchLabel():

    def __init__(self,name="label",path="/Users/zachyamaoka/Documents/de3_audio/data_clip_label/"):

      self.PATH = path
      self.file_stem = name
      file_path =  self.PATH + self.file_stem + '.csv'
      self.file = open(file_path,"a")

    def write(self,audio_file,theta):
        prec = 3
        theta = np.round(theta,prec)
        self.file.write(audio_file + "," + str(theta) + "\n")

    def close(self):
        self.file.close()



class WavWriter():
    def __init__(self,path="/Users/zachyamaoka/Dropbox/de3_audio_data/data_clip/", rate = 44100):

        self.path = path
        self.RATE = rate
        self.WIDTH = 2
        self.CHANNELS = 2

    def save_wav(self, name, data):

        f_name = self.path + name + ".wav"
        wf = wave.open(f_name, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.WIDTH)
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(data))
        wf.close()

#
# file = LabelFile(10)
# for i in range(10):
#     file.write_pos(0.12,0,0)
