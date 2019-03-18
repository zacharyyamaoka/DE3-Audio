import time
import numpy as np
import torch

class AudioLocationNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(2, 96, kernel_size=8, stride=4, padding=1)
        self.conv2 = torch.nn.Conv1d(96, 128, kernel_size=8, stride=4, padding=1)
        self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=1)
        self.conv4 = torch.nn.Conv1d(256, 512, kernel_size=8, stride=4, padding=1)
        self.dense1 = torch.nn.Linear(512*751, 500)
        self.dense2 = torch.nn.Linear(500, 1)

        self.d = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)).view(-1, 512*751)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x

class SoundLocalizer():

    def __init__(self):
        print("SoundLocalizer Init")
        self.model = AudioLocationNN() #instantiate model
        self.model.load_state_dict(torch.load('./../trained_models/40.checkpoint'))

    def locate(self, audio_vec):
        audio_vec = audio_vec.T
        # print("SHHAPE", audio_vec.shape)
        #PUT HAROONS CODE INTO HERE
        model_input=np.expand_dims(audio_vec, 0)
        h = self.model.forward(model_input) #calculate hypothesis
        theta = h.detach().numpy()[showind, 0]

        # time.sleep(2)

        '''radius =  1
        re = audio_vec[0,:]
        le = audio_vec[1,:]

        r_total = np.sum(re)
        l_total = np.sum(le)

        if r_total>l_total:
            theta = np.pi/2

        if r_total<=l_total:
            theta = -np.pi/2

        # theta = np.random.uniform(-np.pi, np.pi)
        theta = -np.pi/2'''
        return theta
