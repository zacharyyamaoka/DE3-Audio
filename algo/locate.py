import time
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "nn_utils"))

from models import *

class SoundLocalizer():

    def __init__(self):
        print("SoundLocalizer Init")
        self.BIN_N = 2
        self.model = AudioLocationNN(self.BIN_N)
        zach_path = "/Users/zachyamaoka/Dropbox/de3_audio_data/trained_model/"
        file = "51.checkpoint"
        self.model.load_state_dict(torch.load(zach_path+file))

    def locate(self, audio_vec):
        audio_vec = audio_vec.T
        # print("SHHAPE", audio_vec.shape)
        #PUT HAROONS CODE INTO HERE
        model_input=np.expand_dims(audio_vec, 0)
        model_input_tensor = torch.from_numpy(model_input)
        model_input_tensor = torch.tensor(model_input_tensor,dtype=torch.float)
        h = self.model.forward(model_input_tensor) #calculate hypothesis
        # print("Hypo: ", h)
        if np.argmax(h.detach().numpy()[0]) ==0:
            theta = np.pi/2
        else:
            theta = 1.5*np.pi
        #
        # if np.random.random() > 0.75:
        #     theta = np.pi/2
        # else:
        #     theta = 1.5*np.pi

        # print("Pred: ", theta)
        # theta = np.pi/2
        # theta = 1.5*np.pi
        confidence = 1
        return theta, confidence
