import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

num_classes = 2

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet3D(nn.Module):
    def __init__(self):
        super(LeNet3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(16, 64, kernel_size=3)
        self.dropout=nn.Dropout(p=0.55)
        self.fc1 = nn.Linear(32768, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32768)
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        x=self.sigmoid(x)
        return x