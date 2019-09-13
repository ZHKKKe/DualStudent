import sys
import math
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable, Function

from third_party.mean_teacher import architectures as mt_arch
from third_party.mean_teacher.utils import export, parameter_count


@export
def cnn3(pretrained=False, **kwargs):
    assert not pretrained
    model = CNN3(**kwargs)
    return model


@export
def cnn13(pretrained=False, **kwargs):
    assert not pretrained
    model = CNN13(**kwargs)
    return model


class GaussianNoise(nn.Module):
    def __init__(self, scale):
        super(GaussianNoise, self).__init__()
        self.scale = scale
    
    def forward(self, x, is_training):
        if not is_training:
            return x

        zeros_ = torch.zeros(x.size()).cuda()
        n = Variable(torch.normal(zeros_, std=1.0).cuda())
        return x + self.scale * n


class CNN3(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN3, self).__init__()
        self.gn = GaussianNoise(0.15)
        self.channels = 32
        self.activation = nn.LeakyReLU(0.1)
        self.conv1 = weight_norm(nn.Conv2d(3, int(self.channels / 2), 3, padding=1))
        self.bn1 = nn.BatchNorm2d(int(self.channels / 2))
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv2 = weight_norm(nn.Conv2d(int(self.channels / 2), self.channels, 3, padding=1))
        self.bn2 = nn.BatchNorm2d(self.channels)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = weight_norm(nn.Conv2d(self.channels, self.channels, 3, padding=1))
        self.bn3 = nn.BatchNorm2d(self.channels)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)
        
        self.fc1 = weight_norm(nn.Linear(self.channels, num_classes))
        self.fc2 = weight_norm(nn.Linear(self.channels, num_classes))

    def forward(self, x, is_training=True):
        x = self.gn(x, is_training)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.mp1(x)

        x = self.activation(self.bn2(self.conv2(x)))
        x = self.mp2(x)
        x = self.drop2(x)

        x = self.activation(self.bn3(self.conv3(x)))
        x = self.ap3(x)
        x = x.view(-1, self.channels)

        return self.fc1(x), self.fc2(x)


class CNN13(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN13, self).__init__()
        self.conv = CNN13_CONV()
        self.fc = CNN13_FC(num_classes=num_classes)

    def forward(self, x, debug=False):
        x = self.conv(x)
        if debug:
            return self.fc(x), x
        else:
            return self.fc(x)


class CNN13_CONV(nn.Module):
    def __init__(self):
        super(CNN13_CONV, self).__init__()
        self.gn = GaussianNoise(0.15)

        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1 = nn.Dropout(0.5)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2 = nn.Dropout(0.5)

        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

    def forward(self, x):
        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        x = self.drop1(x)

        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        x = self.drop2(x)

        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.activation(self.bn3c(self.conv3c(x)))
        x = self.ap3(x)

        x = x.view(-1, 128)
        return x


class CNN13_FC(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN13_FC, self).__init__()
        
        self.fc1 = weight_norm(nn.Linear(128, num_classes))
        self.fc2 = weight_norm(nn.Linear(128, num_classes))

    def forward(self, x):
        return self.fc1(x), self.fc2(x)
