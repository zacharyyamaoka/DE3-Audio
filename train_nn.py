
#Sci Imports
import pandas as pd
import numpy as np
from scipy import signal
import glob
import matplotlib.pyplot as plt

# Torch Imports
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# General Imports
import time
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "nn_utils"))

from nn_util import *
from models import *

#data = AudioLocationDataset(csv="./data_clip_label/label.csv", transform = ToTensor(), use_subset=2100)
# data = AudioLocationDataset(csv="./data_clip_label/label.csv", transform = ToTensor())
train_data = AudioLocationDataset(csv="./data_clip_label/label_train.csv", transform = ToTensor(), use_subset=None)
test_data = AudioLocationDataset(csv="./data_clip_label/label_test.csv", transform = ToTensor(), use_subset=None)

batch_size = 128

train_samples = torch.utils.data.DataLoader(dataset=train_data,
                                              batch_size=batch_size,
                                              shuffle=True)

test_samples = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size)

sample_rate = 96000 #hertz
label_rate = 10 #hertz
chunk_size = 2048 #number of samples to feed to model

lr = 0.0003 #learning rate
regularization = 3e-5
epochs = 50 #number of epochs
model_version = 0



trained_model_path = "./trained_models/"
#trained_model_path = "/Users/zachyamaoka/Dropbox/de3_audio_data/trained_model/"
model = AudioLocationNN() #instantiate model
#model.load_state_dict(torch.load(trained_model_path + str(model_version) + ".checkpoint"))
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularization) #optimizer

def train(epochs):
    lr = 0.003 if epochs<1 else 0.00003
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularization) #optimizer

    model.train()
    #for plotting cost per batch
    costs = []
    movingavg_costs = []
    plt.ion()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    #ax2 = fig.add_subplot(133)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Cost')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    #ax2.set_xlabel('Time')
    #ax2.set_ylabel('Amplitude')

    plt.show()

    for e in range(epochs):
        for i, (x, y) in enumerate(train_samples):
            #print('Audio shape', x.shape, 'Label shape', y.shape)
            #start_ind = np.random.randint(0, xb.shape[2]-chunk_size)
            #end_ind = start_ind+chunk_size
            #x = xb[:, :, start_ind:end_ind]
            #print(x.shape)
            #label_ind = sample2labelId(end_ind, sample_rate, label_rate)
            #y = yb[:, label_ind , :]
            #ylabel = y[:, 1].unsqueeze(dim=0).transpose(0, 1)

            h = model.forward(x) #calculate hypothesis
            #print('H', h.detach().numpy(), '\nLabel', y.detach().numpy())

            # cost = F.mse_loss(h, y) #calculate cost
            #cost = abs_radial_loss(h,y)
            cost = F.cross_entropy(h, y.squeeze())
            #print("COST: ", cost)
            optimizer.zero_grad() #zero gradients
            cost.backward() # calculate derivatives of values of filters
            optimizer.step() #update parameters

            costs.append(cost.item())
            movingavg_costs.append(np.mean(costs[-10:]))
            ax.plot(costs, 'b')
            ax.plot(movingavg_costs, 'g')
            #ax.set_ylim(0, 5)
            showind = np.random.randint(x.shape[0])

            if y.detach().numpy()[showind, 0]==0:
                rhophi1 = [5, np.pi/2]
            else:
                rhophi1 = [5, 1.5*np.pi]
            xy1 = toCartesian(rhophi1)
            # print("H:", h.detach().numpy())
            if np.argmax(h.detach().numpy()[showind]) ==0:
                rhophi2 = [5, np.pi/2]
            else:
                rhophi2 = [5, 1.5*np.pi]
            #rhophi2[0] = np.min([abs(rhophi2[0]), 5])
            xy2 = toCartesian(rhophi2)
            ax1.clear()
            ax1.scatter(np.expand_dims(xy1[0], 0), np.expand_dims(xy1[1], 0), c='g')
            ax1.scatter(np.expand_dims(xy2[0], 0), np.expand_dims(xy2[1], 0), c='b')
            ax1.scatter([[0]], [[0]], c='r', marker='x')
            ax1.set_xlim(-10, 10)
            ax1.set_ylim(-10, 10)

            #ax2.clear()
            #ax2.plot(x.detach().numpy()[showind, 0, :], 'r')
            #ax2.plot(x.detach().numpy()[showind, 1, :], 'b')

            fig.canvas.draw()
            plt.pause(0.00001)
            print('Epoch', e, '\tBatch', i, '\tCost', cost.item(), '\tavgCost', movingavg_costs[-1])
        # if e+1%15==0:
            # torch.save(model.state_dict(), trained_model_path + 'epoch'+str(e)+'.checkpoint')

    costs = np.array(costs)
    plt.close(fig)
    return np.min(costs)

def test(samples):
    model.eval()
    correct=0
    for i, (x, y) in enumerate(samples):
        h = model.forward(x) #calculate hypothesis
        pred = np.argmax(h.detach().numpy(), axis=1)
        y = y.detach().numpy().squeeze()
        c =np.sum(np.equal(pred, y))
        correct+=c
        print('.')
    acc = correct/len(samples.dataset)
    return acc

# learning_rate = 10 ** uniform(-6, 1)
# learning_rate = [1.0,0.1,0.01,0.001,0.0001,0.00001,0.000001]
# loss = []

# for lr in learning_rate:
#     model = AudioLocationNN() #instantiate model
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularization) #optimizer
min_cost = train(epochs)
print("MIN COST: ", min_cost)
#     loss.append(min_cost)

accuracy = test(test_samples)
print('Test accuracy', accuracy)



model_version += epochs
torch.save(model.state_dict(), trained_model_path + str(model_version) + ".checkpoint")
# plt.plot(learning_rate,loss)
# plt.pause(4)
