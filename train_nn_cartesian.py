import pandas as pd
import numpy as np
import glob
from scipy import signal

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "utils"))
from data_utils import *

class AudioLocationDataset(Dataset):
    def __init__(self, root="./data_label/", transform=None):
        self.fnames = glob.glob(root+"*.txt")
        self.root = root
        #self.filenames = self.csv['Filename'].tolist()
        #self.labels = self.csv['Label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        audio, label, rates = load_data_file(n=idx, audio_n_offset=1, label_rate=100, file_stem="data_rec", data_label_path = "./data_label/", data_wav_path = "./../data_wav/")
        audio = audio.T
        label = label[:,:2]
        #cut so they are all the same length
        audio = audio[:, :26146890]
        label = label[:59291, :]

        if self.transform:
            audio, label = self.transform((audio, label))

        return audio, label

def toPolar(xy):
    x = xy[0]
    y = xy[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def toCartesian(rhophi):
    rho = rhophi[0]
    phi = rhophi[1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


class ToTensor():
    def __init__(self):
        pass

    def __call__(self, sample):
        a, l = sample
        return torch.Tensor(a), torch.Tensor(l)

class AudioLocationNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(2, 96, kernel_size=7, stride=2, padding=1)
        self.conv2 = torch.nn.Conv1d(96, 128, kernel_size=7, stride=2, padding=1)
        self.conv3 = torch.nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1)
        self.dense1 = torch.nn.Linear(128*510, 500)
        self.dense2 = torch.nn.Linear(500, 2)

        self.d = torch.nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)).view(-1, 128*510)
        x = F.relu(self.dense1(x))
        x = self.d(x)
        x = self.dense2(x)
        return x          

data = AudioLocationDataset(transform = ToTensor())


batch_size = 3

train_samples = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=batch_size,
                                              shuffle=True)

sample_rate = 44100 #hertz
label_rate = 100 #hertz
chunk_size = 4096 #number of samples to feed to model

lr = 0.0003 #learning rate
epochs = 10 #number of epochs

model = AudioLocationNN() #instantiate model
optimizer = torch.optim.Adam(model.parameters(), lr=lr) #optimizer

def round_down(num, divisor):
    return num - (num%divisor)

def train(epochs):
    #for plotting cost per batch
    costs = []
    plt.ion()
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Cost')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')

    plt.show()

    for e in range(epochs):
        for i, (xb, yb) in enumerate(train_samples):
            #xb, yb = torch.squeeze(xb), torch.squeeze(yb)
            print('Audio shape', xb.shape, 'Label shape', yb.shape)
            for j in range(100):
                #start_ind = np.random.randint(j * sample_rate//label_rate, ((j+1) * sample_rate//label_rate)- chunk_size)
                start_ind = np.random.randint(0, xb.shape[2]-chunk_size)
                end_ind = start_ind+chunk_size
                x = xb[:, :, start_ind:end_ind]
                #print(x.shape)
                label_ind = sample2labelId(end_ind, sample_rate, label_rate)
                y = yb[:, label_ind , :]

                h = model.forward(x) #calculate hypothesis
                print('Pred', h.detach().numpy(), '\nLabel', y.detach().numpy())

                yangles = torch.atan2(y[:, 1], y[:, 0])
                hangles = torch.atan2(h[:, 1], h[:, 0])
                cost = 3*F.mse_loss(hangles, yangles) + F.mse_loss(h, y) #calculate cost

                optimizer.zero_grad() #zero gradients
                cost.backward() # calculate derivatives of values of filters
                optimizer.step() #update parameters

                costs.append(cost.item())
                ax.plot(costs, 'b')
                #ax.set_ylim(0, 100)

                showind = np.random.randint(batch_size)
                predxy = h.detach().numpy()[showind]
                #predxy = 5*predxy/np.linalg.norm(predxy)
                ax1.clear()
                ax1.scatter([y.detach().numpy()[showind, 0]], [y.detach().numpy()[showind, 1]], c='g')
                ax1.scatter([predxy[0]], [predxy[1]], c='b')
                ax1.scatter([[0]], [[0]], c='r', marker='x')
                ax1.set_xlim(-10, 10)
                ax1.set_ylim(-10, 10)

                ax2.clear()
                ax2.plot(x.detach().numpy()[showind, 0, :], 'r')
                ax2.plot(x.detach().numpy()[showind, 1, :], 'b')

                fig.canvas.draw()
                plt.pause(0.001)
                print('Epoch', e, '\tBatch', i, '\tSample', j, '\tCost', cost.item())



train(epochs)
