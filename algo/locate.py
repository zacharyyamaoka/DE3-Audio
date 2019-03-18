import time
import numpy as np
import torch
import torch.nn.functional as F

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

class AudioLocationNNClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(2, 96, kernel_size=8, stride=4, padding=1)
        # conv1.weight.data.fill_(0.01)
        # The same applies for biases:
        #
        # conv1.bias.data.fill_(0.01)
        self.conv2 = torch.nn.Conv1d(96, 128, kernel_size=8, stride=4, padding=1)
        self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=1)
        self.conv4 = torch.nn.Conv1d(256, 512, kernel_size=8, stride=4, padding=1)
        # self.dense1 = torch.nn.Linear(512*751, 500)
        self.dense1 = torch.nn.Linear(512, 500)
        self.dense2 = torch.nn.Linear(500, 2)

        self.d = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x)).view(-1, 512*751)
        x = F.relu(self.conv4(x)).view(-1, 512)
        x = F.relu(self.dense1(x))
        x = F.softmax(self.dense2(x))
        return x


class AudioLocationNNSmall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(2, 96, kernel_size=8, stride=4, padding=1)
        # conv1.weight.data.fill_(0.01)
        # The same applies for biases:
        #
        # conv1.bias.data.fill_(0.01)
        self.conv2 = torch.nn.Conv1d(96, 128, kernel_size=8, stride=4, padding=1)
        self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=1)
        self.conv4 = torch.nn.Conv1d(256, 512, kernel_size=8, stride=4, padding=1)
        # self.dense1 = torch.nn.Linear(512*751, 500)
        self.dense1 = torch.nn.Linear(512, 500)
        self.dense2 = torch.nn.Linear(500, 1)

        self.d = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x)).view(-1, 512*751)
        x = F.relu(self.conv4(x)).view(-1, 512)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x

class SoundLocalizer():

    def __init__(self):
        print("SoundLocalizer Init")
        # self.model = AudioLocationNN() #instantiate model
        # self.model = AudioLocationNNSmall() #instantiate model
        self.model = AudioLocationNNClass() #instantiate model

        zach_path = "/Users/zachyamaoka/Dropbox/de3_audio_data/trained_model/"
        file = "198.checkpoint"
        self.model.load_state_dict(torch.load(zach_path+file))

    def locate(self, audio_vec):
        audio_vec = audio_vec.T
        # print("SHHAPE", audio_vec.shape)
        #PUT HAROONS CODE INTO HERE
        model_input=np.expand_dims(audio_vec, 0)
        model_input_tensor = torch.from_numpy(model_input)
        model_input_tensor = torch.tensor(model_input_tensor,dtype=torch.float)
        h = self.model.forward(model_input_tensor) #calculate hypothesis
        # print("HHHHHHH")
        # print(h)
        if np.argmax(h.detach().numpy()[0]) ==0:
            theta = np.pi/2
        else:
            theta = 1.5*np.pi


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
