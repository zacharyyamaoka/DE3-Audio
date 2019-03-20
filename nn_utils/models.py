import torch
import torch.nn.functional as F

class AudioLocationNN(torch.nn.Module):
    def __init__(self, bins):
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
        assert bins % 2 == 0 #must have an equal number of bins
        self.dense2 = torch.nn.Linear(500, bins)

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
