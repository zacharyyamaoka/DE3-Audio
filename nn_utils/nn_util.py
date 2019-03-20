
#Imports
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import librosa

import torch


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

def train_test_val_split(csv="./data_clip_label/label.csv", save_loc='./data_clip_label/'):
    csv = pd.read_csv(csv)
    filenames = csv['Filename'].tolist()
    labels = csv['Label'].tolist()
    all_i = np.arange(len(filenames))
    all_i = np.random.choice(all_i, len(all_i), replace=False)

    train=csv.sample(frac=0.8)
    testval=csv.drop(train.index)
    val=testval.sample(frac=0.5)
    test=testval.drop(val.index)

    print(type(train), '\n', train.head(), '\n len', len(train))
    print(type(val), '\n', val.head(), '\n len', len(val))
    print(type(test), '\n', test.head(), '\n len', len(test))

    train.to_csv(save_loc+"label_train.csv", index=False)
    test.to_csv(save_loc+"label_val.csv", index=False)
    test.to_csv(save_loc+"label_test.csv", index=False)

class ToTensor():
    def __init__(self):
        pass

    def __call__(self, sample):
        a, l = sample
        return torch.Tensor(a), torch.LongTensor(l)


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

def round_down(num, divisor):
    return num - (num%divisor)

def radial_loss(h, y):
    x = torch.abs(h.sub(y))
    x = torch.remainder(x, np.pi)
    x = torch.mean(x)
    return x

def get_bins(n):
    """ Splits heading into a discrete number of bins"""
    assert n % 2==0 # Onlys works for symetric bin numbers. Only symetric sub divsions are meaningful for the dummy headself.

    bin_size = (2 * np.pi)/n

    # Not tested, changes so bin go along middle.
    # how front how back......, saying front or back is easier.

    if n == 2:
        bin_offset = 0 # left right case
    else:
        bin_offset = bin_size/2

    start = 0
    theta = start + bin_offset
    bins = []
    for i in range(n):

        bins.append(theta)
        theta += bin_size

    return bins

def get_theta_quad(theta, n): #Rounds to center of quadrant

    #floors theta based on the number of bins

    


def segment_data(theta,bins):
    """segments polar data into respective bin and return ind, here 0 corresponds to the first bin. Assums data is between 0 and 2pi"""
    #bins of the from end
    n = len(bins)

    #find smallest starting value.


    for i in np.arange(n):
        j = i + 1
        if j > n - 1:
            return n - 1 #edge case
        if (theta >= bins[i]) and (theta <= bins[j]):
            return i #return the ind



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
