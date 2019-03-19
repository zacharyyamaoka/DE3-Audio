import pandas as pd
import numpy as np
from scipy import signal
import librosa
import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import time
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "utils"))
from data_utils import *

class AudioLocationDataset(Dataset):
    def __init__(self, root="./../data_clip/", csv="./data_clip_label/label.csv", transform=None, use_subset=None):
        self.root = root
        self.csv = pd.read_csv(csv)
        if use_subset is not None:
            self.filenames = self.csv['Filename'].tolist()[:use_subset]
            self.labels = self.csv['Label'].tolist()[:use_subset]
        else:
            self.filenames = self.csv['Filename'].tolist()
            self.labels = self.csv['Label'].tolist()

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        #audio, label, rates = load_data_file(n=idx, audio_n_offset=0, label_rate=10, file_stem="real_rec_", data_label_path = "./data_real_label/", data_wav_path = "./../data_real_wav/")
        fname = self.filenames[idx]
        label = self.labels[idx]
        path = self.root + fname
        audio, sample_rate = librosa.core.load(path, sr=96000, mono=False)
        #print(audio.shape)
        #print([label])
        #label = label[:, :2]
        #label = np.expand_dims(label, 1)

        #cut so they are all the same length
        # audio = audio[:, :192512]  #26146890 for synthetic

        # Take random 0.1 sample
        rate = 96000
        dur = 0.005
        chunk = int(rate*dur)

        max_rand_ind = 192512 - chunk - 1
        min_rand_ind = 0
        start = int(np.random.uniform(min_rand_ind,max_rand_ind))
        # print(start, start+chunk)
        audio = audio[:, start:(start+chunk)]

        #center data
        mean = np.mean(audio)
        audio -= mean

        #normalize

        max = np.max(np.abs(audio))
        audio /= max
        #label = label[:5995, :] #59291 for synthetic
        if label<np.pi:
            label=[0]
        else:
            label=[1]
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

def test_train_split(csv="./data_clip_label/label_OG.csv", save_loc='./data_clip_label/'):
    csv = pd.read_csv(csv)
    filenames = csv['Filename'].tolist()
    labels = csv['Label'].tolist()
    all_i = np.arange(len(filenames))
    all_i = np.random.choice(all_i, len(all_i), replace=False)
    
    train=csv.sample(frac=0.8)
    test=csv.drop(train.index)

    print(type(train), '\n', train.head(), '\n len', len(train))
    print(type(test), '\n', test.head(), '\n len', len(test))

    train.to_csv(save_loc+"label_train.csv", index=False)
    test.to_csv(save_loc+"label_test.csv", index=False)
    
class ToTensor():
    def __init__(self):
        pass

    def __call__(self, sample):
        a, l = sample
        return torch.Tensor(a), torch.LongTensor(l)

class AudioLocationNN(torch.nn.Module):
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


def abs_radial_loss(h,y):
    global batch_size

    x = torch.abs(h.sub(y))
    x = torch.abs(x - np.pi)
    x = np.pi - x
    # print(x)
    # showind = np.random.randint(x.shape[0])
    # label = y.detach().numpy()[showind, 0]
    # pred = h.detach().numpy()[showind, 0]
    # x_ = x.detach().numpy()[showind, 0]
    # print("label: ", np.rad2deg(label), "pred: ", np.rad2deg(pred), "diff: ", np.rad2deg(x_))
    # time.sleep(3)
    # x = x * x #square difference
    x = torch.abs(x) # must be positive
    x = torch.sum(x)
    x = x/batch_size

    return x

trained_model_path = "./trained_models/"
#trained_model_path = "/Users/zachyamaoka/Dropbox/de3_audio_data/trained_model/"
model = AudioLocationNN() #instantiate model
#model.load_state_dict(torch.load(trained_model_path + str(model_version) + ".checkpoint"))
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularization) #optimizer

def round_down(num, divisor):
    return num - (num%divisor)

def radial_loss(h, y):
    x = torch.abs(h.sub(y))
    x = torch.remainder(x, np.pi)
    x = torch.mean(x)
    return x


# def abs_radial_loss(h,y):
#     global batch_size
#     # h = torch.remainder(h, np.pi) #
#     x = torch.abs(h.sub(y))
#     x = torch.abs(x - np.pi)
#     x = np.pi - x
#     x = x * x #square value
#
#     x = torch.sum(x) # average over batch
#     x = x / batch_size


    #return x

def train(epochs):
    model.train()
    #for plotting cost per batch
    costs = []
    movingavg_costs = []
    plt.ion()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Cost')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')

    plt.show()

    for e in range(epochs):
        for i, (x, y) in enumerate(train_samples):
            print('Audio shape', x.shape, 'Label shape', y.shape)
            #start_ind = np.random.randint(0, xb.shape[2]-chunk_size)
            #end_ind = start_ind+chunk_size
            #x = xb[:, :, start_ind:end_ind]
            #print(x.shape)
            #label_ind = sample2labelId(end_ind, sample_rate, label_rate)
            #y = yb[:, label_ind , :]
            #ylabel = y[:, 1].unsqueeze(dim=0).transpose(0, 1)

            h = model.forward(x) #calculate hypothesis
            print('H', h.detach().numpy(), '\nLabel', y.detach().numpy())

            # cost = F.mse_loss(h, y) #calculate cost
            #cost = abs_radial_loss(h,y)
            cost = F.cross_entropy(h, y.squeeze())
            print("COST: ", cost)
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

            ax2.clear()
            ax2.plot(x.detach().numpy()[showind, 0, :], 'r')
            ax2.plot(x.detach().numpy()[showind, 1, :], 'b')

            fig.canvas.draw()
            plt.pause(0.00001)
            print('Epoch', e, '\tBatch', i, '\tCost', cost.item(), '\tavgCost', movingavg_costs[-1])
        # if e+1%15==0:
            # torch.save(model.state_dict(), trained_model_path + 'epoch'+str(e)+'.checkpoint')

    costs = np.array(costs)
    plt.close(fig)
    return np.min(costs)

def test():
    model.eval()
    correct=0
    for i, (x, y) in enumerate(test_samples):
        print('Audio shape', x.shape, 'Label shape', y.shape)

        h = model.forward(x) #calculate hypothesis
        pred = np.argmax(h.detach().numpy(), axis=1)
        y = y.detach().numpy().squeeze()
        c =np.sum(np.equal(pred, y))
        correct+=c
    acc = correct/len(test_data)
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

accuracy = test()
print('Test accuracy', accuracy)



model_version += epochs
torch.save(model.state_dict(), trained_model_path + str(model_version) + ".checkpoint")
# plt.plot(learning_rate,loss)
# plt.pause(4)
