import torch
import numpy as np
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
from tqdm import tqdm
import os
import csv
from read_data import mix_index,max_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_3d_size = max_index-mix_index

def test_model(model,data_test,conte,filename):
    file=open(filename,'w',newline='')
    csv_write=csv.writer(file)
    csv_write.writerow(["Id","Predicted"])
    i=0
    model.eval()
    for data in tqdm(data_test):
        voxel=np.array(data)
        voxel=torch.tensor(voxel).float()
        voxel=voxel.view(-1,1,my_3d_size,my_3d_size,my_3d_size)
        voxel=voxel.to(device)
        out=model(voxel)
        a=out.tolist()
        row=[conte[i+1][0],a[0][1]/(a[0][0]+a[0][1])]
        csv_write.writerow(row)
        i+=1
    file.close()