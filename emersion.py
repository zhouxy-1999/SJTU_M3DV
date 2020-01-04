import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from read_data import get_data
from mylenet3d import LeNet3D
from my_test import test_model
from my_train import train_model

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == "__main__":
    model=torch.load('my_model.pth')
    model1=model.to(device)
    data_test,te_label,conte,data_tevoxel=get_data('test')
    filename="emersion.csv"
    test_model(model1,data_test,conte,filename)
