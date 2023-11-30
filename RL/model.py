import torch
import torch.nn as nn
import torch.nn.functional as F
from .hparams import HyperParams as hp


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs1, num_outputs2):
        self.num_inputs = num_inputs
        self.num_outputs1 = num_outputs1
        self.num_outputs2 = num_outputs2
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3_1 = nn.Linear(hp.hidden, self.num_outputs1)
        self.fc3_2 = nn.Linear(hp.hidden, self.num_outputs2)

        self.fc3_1.weight.data.mul_(0.1)
        self.fc3_1.bias.data.mul_(0.0)
        self.fc3_2.weight.data.mul_(0.1)
        self.fc3_2.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mu1 = self.fc3_1(x)
        logstd1 = torch.zeros_like(mu1)
        std1 = torch.exp(logstd1)
        mu2 = self.fc3_1(x)
        logstd2 = torch.zeros_like(mu2)
        std2 = torch.exp(logstd2)
        return mu1, std1, logstd1, mu2, std2, logstd2


class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        v = self.fc3(x)
        return v