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
    model= LeNet3D()
    model1=model.to(device)

    #  train_val.csv，test.csv，训练集与测试集文件与.py文件放在同一目录下
    data_train,data_label,cont,_=get_data('train_val')    
    data_test,_,conte,_=get_data('test')
    filename="result.csv"                                 # 最后预测的结果保存在.py同目录下的result.csv中

    model1,x_axi,loss=train_model(model1,data_train,data_label,cont,10)
    test_model(model1,data_test,conte,filename)
    x=np.linspace(0,x_axi,x_axi,endpoint=True)

    # 打印出loss
    plt.plot(x,loss,'r')
    plt.show()
