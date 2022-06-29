import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,dim_in,dim_hidden,dim_out):
        super(MLP,self).__init__()
        self.layer_hidden = nn.Linear(dim_in,dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_out = nn.Linear(dim_hidden,dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,X):
        X = X.view(-1,X.shape[1] * X.shape[-2] * X.shape[-1])
        X = self.layer_hidden(X)
        X = self.dropout(X)
        X = self.relu(X)
        X = self.layer_out(X)
        return self.softmax(X)

class CNNMnist(nn.Module):
    def __init__(self,args):
        super(CNNMnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 32, kernel_size=(5, 5)),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.MaxPool2d(2))
        self.full_connection = nn.Linear(64 * 4 * 4, args.num_classes)

    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
        X = self.full_connection(X)
        X = F.relu(X)
        return F.softmax(X)
    

class CNNFashion_Mnist(nn.Module):
    def __init__(self,args):
        super(CNNFashion_Mnist,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(args.num_channels, 32, kernel_size=(5, 5)),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.MaxPool2d(2))
        self.full_connection = nn.Linear(64 * 4 * 4, args.num_classes)

    def forward(self,X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
        X = self.full_connection(X)
        X = F.relu(X)
        return F.softmax(X)
    

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128 * 4 * 4, 1028)
        self.fc2 = nn.Linear(1028, args.num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

