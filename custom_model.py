import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 3), padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(2, 2), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(96)
        
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(2, 2), stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=160, kernel_size=(2, 2), stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(160)
        
        self.conv6 = nn.Conv2d(in_channels=160, out_channels=192, kernel_size=(2, 2), stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(192)
        
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(192, 100)
        
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 50)
        
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(50, 25)
        
        self.dropout4 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(25, 1)
        
    def forward(self, x):
        # Forward pass through convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.bn6(self.conv6(x)))
        x = torch.amax(x, dim=(-2, -1))  # Global max pooling
        
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        
        x = self.dropout3(x)
        x = F.relu(self.fc3(x))
        
        x = self.dropout4(x)
        x = torch.sigmoid(self.fc4(x))
        
        return x
