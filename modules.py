import torch.nn as nn
import torch
import os

# GLOBAL VARS:
CHANNELS = 3
FEATURES = 16
IMG_SHAPE = 208
RANDOM_VEC_SIZE = 256

class Discriminator(nn.Module):
    def __init__(self, channels = CHANNELS, features = FEATURES):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = channels, out_channels = features, kernel_size = 16, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = features, out_channels = features * 2, kernel_size = 16, stride = 2, padding =1)
        self.bn1 = nn.BatchNorm2d(num_features = features * 2)
        self.mp1 = nn.MaxPool2d(kernel_size = (2, 2))
        self.conv3 = nn.Conv2d(in_channels = features * 2, out_channels = features * 4, kernel_size = 8, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = features * 4, out_channels = features * 8, kernel_size = 8, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(num_features = features * 8)
        self.conv5 = nn.Conv2d(in_channels = features * 8, out_channels = features * 16, kernel_size = 4, stride = 2, padding = 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear( 16, 1)
            
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.bn1(self.conv2(x)))
        x = self.mp1(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.bn2(self.conv4(x)))
        x = torch.relu(self.conv5(x))
        x = x.view(-1, 256)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.output(x))
        return x
    
    
class Generator(nn.Module):
    def __init__(self, noise_size = RANDOM_VEC_SIZE, channels = CHANNELS, features = FEATURES):
        super().__init__()
        self.convm2 = nn.ConvTranspose2d(in_channels = noise_size, out_channels = features*256, kernel_size = 4, stride = 1, padding = 0)
        self.bnm2 = nn.BatchNorm2d(features * 256)

        self.convm1 = nn.ConvTranspose2d(in_channels = features * 256, out_channels = features*64, kernel_size = 4, stride = 1, padding = 0)
        self.bnm1 = nn.BatchNorm2d(features * 64)

        self.conv0 = nn.ConvTranspose2d(in_channels = features * 64, out_channels = features*32, kernel_size = 4, stride = 1, padding = 0)
        self.bn0 = nn.BatchNorm2d(features * 32)

        self.conv1 = nn.ConvTranspose2d(in_channels = features * 32, out_channels = features*16, kernel_size = 4, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(features * 16)
        # self.relu1 = nn.ReLU(),

        self.conv2 = nn.ConvTranspose2d(in_channels = features * 16, out_channels = features * 8, kernel_size = 4, stride = 2, padding = 1)
        self.bn2 =   nn.BatchNorm2d(num_features = features * 8)
        # self.relu2 =   nn.ReLU(),

        self.conv3 =nn.ConvTranspose2d(in_channels = features * 8, out_channels = features * 4, kernel_size = 4, stride = 2, padding = 1)
        self.bn3 =    nn.BatchNorm2d(num_features = features * 4)
        # self.relu3 =  nn.ReLU(),

        self.conv4 = nn.ConvTranspose2d(in_channels = features * 4, out_channels = features * 2, kernel_size = 4, stride = 2, padding = 1)
        self.bn4 = nn.BatchNorm2d(num_features = features * 2)
        # self.relu4 = nn.ReLU(),

        self.conv5 = nn.ConvTranspose2d(in_channels = features * 2, out_channels = channels, kernel_size = 4, stride = 2, padding = 1)
        # self.tn = nn.Tanh()
        
        
    def forward(self, x):
        x = torch.relu(self.bnm2(self.convm2(x)))
        x = torch.relu(self.bnm1(self.convm1(x)))
        x = torch.relu(self.bn0(self.conv0(x)))
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.tanh(self.conv5(x))
        return x