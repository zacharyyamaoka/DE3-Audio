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
        audio, label, rates = load_data_file(n=idx, audio_n_offset=1, label_rate=100)
        audio = audio.T
        label = label[:, :2]
        label = np.apply_along_axis(toPolar, 1, label)
        #cut so they are all the same length
        audio = audio[:, :26146890]
        label = label[:59291, :]
        frequencies, times, spec1 = signal.spectrogram(audio[1, :], 44100)
        frequencies, times, spec2 = signal.spectrogram(audio[1, :], 44100)
        spectrogram = np.vstack((spec1, spec2))

        if self.transform:
            spectrogram, label = self.transform((spectrogram, label))

        return spectrogram, label

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
        self.conv1 = torch.nn.Conv1d(258, 128, kernel_size=4, stride=2, padding=1)
        #self.conv2 = torch.nn.Conv1d(2, 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = torch.nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.dense1 = torch.nn.Linear(128, 1)
    def forward(self, x):
        x = F.relu(self.conv1(x)).view(-1, 128)
        #print(x.shape)
        #x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x)).view(-1, 256)
        x = self.dense1(x)
        return x

'''
audio, label, rates = load_data_file(n=2, audio_n_offset=1, label_rate=100)
audio = audio.T
label = label[:, :2]
label = np.apply_along_axis(toPolar, 1, label)
#cut so they are all the same length
audio = audio[:, :26146890]
label = label[:59291, :]
samples = audio[1, :]
print(samples.shape)
frequencies, times, spectrogram = signal.spectrogram(samples, 44100)
print(frequencies[:10], frequencies.shape)
print(times[:10], times.shape)
print(spectrogram.shape)
plt.figure()
plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram[:, :1000])
plt.show()
gg
'''

        

data = AudioLocationDataset(transform = ToTensor())


batch_size = 3

train_samples = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=batch_size,
                                              shuffle=True)

sample_rate = 196.875 #hertz
label_rate = 100 #hertz
chunk_size = 2 #number of samples to feed to model

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
    ax.set_xlabel('Sample')
    ax.set_ylabel('Cost')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

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

                ylabel = y[:, 1].unsqueeze(dim=0).transpose(0, 1)

                h = model.forward(x) #calculate hypothesis
                print('Pred', h.detach().numpy(), '\nLabel', ylabel.detach().numpy())

                cost = F.mse_loss(h, ylabel) #calculate cost

                optimizer.zero_grad() #zero gradients
                cost.backward() # calculate derivatives of values of filters
                optimizer.step() #update parameters

                costs.append(cost.item())
                ax.plot(costs, 'b')
                ax.set_ylim(0, 100)

                showind = np.random.randint(batch_size)

                rhophi1 = [y.detach().numpy()[showind, 0], y.detach().numpy()[showind, 1]]
                xy1 = toCartesian(rhophi1)
                rhophi2 = [5, h.detach().numpy()[showind, 0]] #h.detach().numpy()[0, 0]
                #rhophi2[0] = np.min([abs(rhophi2[0]), 5])
                xy2 = toCartesian(rhophi2)
                ax1.clear()
                ax1.scatter(np.expand_dims(xy1[0], 0), np.expand_dims(xy1[1], 0), c='g')
                ax1.scatter(np.expand_dims(xy2[0], 0), np.expand_dims(xy2[1], 0), c='b')
                ax1.scatter([[0]], [[0]], c='r', marker='x')
                ax1.set_xlim(-10, 10)
                ax1.set_ylim(-10, 10)

                fig.canvas.draw()
                plt.pause(0.001)
                print('Epoch', e, '\tBatch', i, '\tSample', j, '\tCost', cost.item())



train(epochs)
