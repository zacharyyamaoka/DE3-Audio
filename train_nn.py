import pandas as pd
import numpy as np
import glob

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
        audio, label, rates = load_data_file(n=idx)
        audio = audio.T
        label = label[:, :2]
        print(label)
        #for i in range(len(label)):
            #label[i] = toPolar(label[i])
        label = np.apply_along_axis(toPolar, 1, label)
        print(label)

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
        self.conv1 = torch.nn.Conv1d(2, 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = torch.nn.Conv1d(2, 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = torch.nn.Conv1d(2, 2, kernel_size=4, stride=2, padding=1)
        self.dense1 = torch.nn.Linear(1024, 2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)).view(-1, 1024)
        x = self.dense1(x)
        return x


data = AudioLocationDataset(transform = ToTensor())

batch_size = 1

train_samples = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=batch_size,
                                              shuffle=True)

sample_rate = 44100 #hertz
label_rate = 1 #hertz
chunk_size = 4096 #number of samples to feed to model

lr = 0.001 #learning rate
epochs = 10 #number of epochs

model = AudioLocationNN() #instantiate model
optimizer = torch.optim.Adam(model.parameters(), lr=lr) #optimizer

def round_down(num, divisor):
    return num - (num%divisor)

def train(epochs):
    #for plotting cost per batch
    costs = []
    plt.ion()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    plt.show()
    ax.set_xlabel('Sample')
    ax.set_ylabel('Cost')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    for e in range(epochs):
        for i, (xb, yb) in enumerate(train_samples):
            #xb, yb = torch.squeeze(xb), torch.squeeze(yb)
            #print(xb.shape, yb.shape)
            for j in range(yb.shape[1]):
                start_ind = np.random.randint(j * sample_rate//label_rate, ((j+1) * sample_rate//label_rate)- chunk_size)
                x, y = xb[0, :, start_ind:start_ind+chunk_size].unsqueeze(dim=0), yb[0, j].unsqueeze(dim=0)

                h = model.forward(x) #calculate hypothesis
                print('Pred', h.detach().numpy(), 'Label', y.detach().numpy())

                cost = F.mse_loss(h, y) #calculate cost

                optimizer.zero_grad() #zero gradients
                cost.backward() # calculate derivatives of values of filters
                optimizer.step() #update parameters

                costs.append(cost.item())
                ax.plot(costs, 'b')
                # ax.set_ylim(0, 50)

                rhophi1 = [y.detach().numpy()[0, 0], y.detach().numpy()[0, 1]]
                xy1 = toCartesian(rhophi1)
                rhophi2 = [h.detach().numpy()[0, 0], h.detach().numpy()[0, 1]]
                xy2 = toCartesian(rhophi2)
                ax1.clear()
                ax1.scatter(np.expand_dims(xy1[0], 0), np.expand_dims(xy1[1], 0), c='g')
                ax1.scatter(np.expand_dims(xy2[0], 0), np.expand_dims(xy2[1], 0), c='b')
                ax1.set_xlim(-5, 5)
                ax1.set_ylim(-5, 5)

                fig.canvas.draw()
                plt.pause(0.001)
                print('Epoch', e, '\tAudioclip', i, '\tSample', j, '\tCost', cost.item())



train(epochs)
