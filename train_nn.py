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


batch_size = 128
BIN_N = 2

#train_test_val_split(csv='./data_clip_label/label_full_mon.csv')

train_data = AudioLocationDataset(csv="./data_clip_label/label_train.csv", transform = ToTensor(), use_subset=None, num_bin=BIN_N)
val_data = AudioLocationDataset(csv="./data_clip_label/label_val.csv", transform = ToTensor(), use_subset=None, num_bin=BIN_N)
test_data = AudioLocationDataset(csv="./data_clip_label/label_test.csv", transform = ToTensor(), use_subset=None, num_bin=BIN_N)

train_samples = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=batch_size,
                                            shuffle=True)

val_samples = torch.utils.data.DataLoader(dataset=val_data,
                                              batch_size=batch_size)

test_samples = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=batch_size)

def cosine_curve_schedule(max_lr, epochs, decay_param=3):
    lr_schedule = max_lr*np.cos((np.arange(epochs)/epochs)*0.5*np.pi)
    lr_schedule = np.exp(decay_param*(lr_schedule/max_lr))*lr_schedule/np.exp(decay_param)
    return lr_schedule

# PARAMETERS
sample_rate = 96000 #hertz
label_rate = 10 #hertz
chunk_size = 2048 #number of samples to feed to model

epochs = 40 #number of epochs
max_lr = 0.01 #learning rate
lr_schedule = cosine_curve_schedule(max_lr, epochs, decay_param=6)
regularization = 3e-5

model_version = 51

#print(lr_schedule)
#plt.plot(lr_schedule)
#plt.show()
#gg


trained_model_path = "./trained_models/"
#trained_model_path = "/Users/zachyamaoka/Dropbox/de3_audio_data/trained_model/"
model = AudioLocationNN(BIN_N) #instantiate model
model.load_state_dict(torch.load(trained_model_path + str(model_version) + ".checkpoint"))
optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=regularization) #optimizer

def train(epochs, lr_schedule=None):

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
        if lr_schedule is not None:
            clr = lr_schedule[e]
            optimizer = torch.optim.Adam(model.parameters(), lr=clr, weight_decay=regularization) #optimizer
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=regularization) #optimizer
        for i, (x, y) in enumerate(train_samples):

            h = model.forward(x) #calculate hypothesis

            cost = F.cross_entropy(h, y.squeeze())

            optimizer.zero_grad() #zero gradients
            cost.backward() # calculate derivatives of values of filters
            optimizer.step() #update parameters

            costs.append(cost.item())
            movingavg_costs.append(np.mean(costs[-10:]))
            ax.plot(costs, 'b')
            ax.plot(movingavg_costs, 'g')
            #ax.set_ylim(0, 5)
            showind = np.random.randint(x.shape[0])

            # if y.detach().numpy()[showind, 0]==0:
            #     rhophi1 = [5, np.pi/2]
            # else:
            #     rhophi1 = [5, 1.5*np.pi]
            #print(h.detach().numpy())
            pred_label = np.argmax(h.detach().numpy()[showind])
            pred_actual = y.detach().numpy()[showind]

            theta_pred = get_theta_quad(pred_label, BIN_N)
            theta_actual = get_theta_quad(pred_actual, BIN_N)

            rhophi1 = [5, theta_pred]
            rhophi2 = [5, theta_actual]

            xy1 = toCartesian(rhophi1)
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
            print('Epoch', e, '\tBatch', i, '\tCost', cost.item(), '\tavgCost', movingavg_costs[-1], '\tLearning rate', clr)
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
        print(h)
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
#min_cost = train(epochs, lr_schedule)
#print("MIN COST: ", min_cost)
#     loss.append(min_cost)

accuracy = test(test_samples)
print('Test accuracy', accuracy)

#model_version += epochs
#torch.save(model.state_dict(), trained_model_path + str(model_version) + ".checkpoint")
# plt.plot(learning_rate,loss)
# plt.pause(4)
