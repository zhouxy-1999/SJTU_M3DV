import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch import nn,optim
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
from numpy import random
import time
import os
import csv
from read_data import mix_index,max_index

num_classes = 2
NUM_EPOCHS = 15
BATCH_SIZE = 12

my_3d_size = max_index-mix_index

normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mixup(x,y,alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    result=Variable(lam*x+(1.-lam)*y)
    return result,lam

def cutout(tube,n_holes,length):
    d = tube.size(2)
    h = tube.size(3)
    w = tube.size(4)

    mask = np.ones((d,h, w), np.float32)

    for n in range(n_holes):
        mask_d = np.random.randint(d-length)
        mask_h = np.random.randint(h-length)
        mask_w = np.random.randint(w-length)

        mask[mask_d:mask_d+length, mask_h:mask_h+length, mask_w:mask_w+length] = 0.

    mask = torch.from_numpy(mask) 
    mask = mask.expand_as(tube)
    tube = tube * mask

    return tube

# 数据扩增：将数据分别沿x，y，z轴翻转180度
def transform(my3D_matrix,my_label):
    d , h , w = my3D_matrix.shape
    new3D_matrix = np.zeros((d,h,w))
    if my_label == 0:
        for i in range(d):
            new3D_matrix[d-i-1][:][:] = my3D_matrix[i][:][:]
    if my_label == 1:
        for i in range(d):
            for j in range(h):
                new3D_matrix[i][h-1-j][:] = my3D_matrix[i][j][:]
    if my_label == 2:
        for i in range(d):
            for j in range(h):
                for k in range(w):
                    new3D_matrix[i][j][w-1-k] = my3D_matrix[i][j][k]
    return new3D_matrix

#------------------------------------------------- train -----------------------------------------------------------------------
def train_model(model,data_train,data_label,c,model_num):
    model.train()
    criterion = nn.BCELoss().cuda()

    #优化器
    optimizer = torch.optim.Adam(model.parameters())
    #损失函数
    all_loss=[] ; x_axi=0 ; a=[] ; brea_num=0 ; tmp_num=0.

    onebatch_datasize=int(465/BATCH_SIZE+1)
    all_batch=1
    tmp_loss=0.
    for nu in range(onebatch_datasize):
        a.append(nu*BATCH_SIZE)
    for epoch in range(NUM_EPOCHS): 
        wez=np.arange(465)
        weizhi1=wez.tolist(); random.shuffle(weizhi1)
        data_train1=[] ; data_label1=[]
        for dii in range(465):
            data_train1.append(data_train[weizhi1[dii]])
            data_label1.append(data_label[weizhi1[dii]])
        data_train=data_train1
        data_label=data_label1 
        correct=0.0
        total=0.0
        i=0
        for tmp in tqdm(a):
            fen_loss=torch.tensor(0).to(device).float()
            weizhi=[]
            if tmp==a[onebatch_datasize-1]:
                zhege=data_train[tmp:465]
                tmp_label=data_label[tmp:465]
            else:
                zhege=data_train[tmp:tmp+BATCH_SIZE]
                tmp_label=data_label[tmp:tmp+BATCH_SIZE]
            for san in range(len(zhege)):
                weizhi.append(san)
            random.shuffle(weizhi)
            #TODO:nothing
            for san in range(len(zhege)):
                x=np.array(zhege[san]) ;x=torch.tensor(x).float() ; x=x.view(-1,1,my_3d_size,my_3d_size,my_3d_size) ; x=x.to(device)
                x_t=tmp_label[san] ; x_la=torch.tensor([[1-float(x_t),float(x_t)]]);x_la=x_la.to(device) 
                out_x=model(x);pre_x=torch.argmax(out_x,1)
                total+=1 ; correct+=(pre_x==x_t).sum().item()
                loss=criterion(out_x,x_la)
                fen_loss+=loss
            loss=fen_loss/len(zhege)
            tmp_loss+=loss.float()
            if all_batch % 18==0:
                tmp_loss=tmp_loss/18
                all_loss.append(tmp_loss)
                tmp_loss=0.
                x_axi+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_batch+=1
            fen_loss=torch.tensor(0).to(device).float()
            '''
            for m in range(1):
                for san in range(len(zhege)):
                    x_tmp=np.array(zhege[san]) ;x=transform(x_tmp,m);x=torch.tensor(x).float() ; x=x.view(-1,1,my_3d_size,my_3d_size,my_3d_size) ; x=x.to(device)
                    x_t=tmp_label[san] ; x_la=torch.tensor([[1-float(x_t),float(x_t)]]);x_la=x_la.to(device) 
                    out_x=model(x);pre_x=torch.argmax(out_x,1)
                    total+=1 ; correct+=(pre_x==x_t).sum().item()
                    loss=criterion(out_x,x_la)
                    fen_loss+=loss
                loss=fen_loss/len(zhege)
                tmp_loss+=loss.float()
                if all_batch % 18==0:
                    tmp_loss=tmp_loss/18
                    all_loss.append(tmp_loss)
                    tmp_loss=0.
                    x_axi+=1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                all_batch+=1
                fen_loss=torch.tensor(0).to(device).float()
            '''
            #TODO:cutout
            for san in range(len(zhege)):
                x=np.array(zhege[san]) ;x=torch.tensor(x).float() ; x=x.view(-1,1,my_3d_size,my_3d_size,my_3d_size) 
                x=cutout(x,4,6) #TODO:Change the cutout size and batchs-------
                x=x.to(device)
                x_t=tmp_label[san] ; x_la=torch.tensor([[1-float(x_t),float(x_t)]]);x_la=x_la.to(device) 
                out_x=model(x);pre_x=torch.argmax(out_x,1)
                total+=1 ; correct+=(pre_x==x_t).sum().item()
                loss=criterion(out_x,x_la)
                fen_loss+=loss
            loss=fen_loss/len(zhege)
            tmp_loss+=loss.float()
            if all_batch % 18==0:
                tmp_loss=tmp_loss/18
                all_loss.append(tmp_loss)
                tmp_loss=0.
                x_axi+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_batch+=1
            fen_loss=torch.tensor(0).to(device).float()
            
            #TODO:mixup
            for san in range(len(zhege)):
                x=np.array(zhege[weizhi[san]]) ; y=np.array(zhege[(weizhi[san]+1)%len(zhege)])
                x=torch.tensor(x).float() ; x=x.view(-1,1,my_3d_size,my_3d_size,my_3d_size) ; x=x.to(device)
                y=torch.tensor(y).float() ; y=y.view(-1,1,my_3d_size,my_3d_size,my_3d_size) ; y=y.to(device)

                x_t=tmp_label[weizhi[san]] ; y_t=tmp_label[(weizhi[san]+1)%len(zhege)]
                vox1,lam=mixup(x,y)
                x_la=torch.tensor([[1-float(x_t),float(x_t)]]) ; y_la=torch.tensor([[1-float(y_t),float(y_t)]])
                x_la=x_la.to(device) ; y_la=y_la.to(device)
                out=model(vox1);out_x=model(x) ; out_y=model(y)
                pre_x=torch.argmax(out_x,1) ; pre_y=torch.argmax(out_y,1)
                total+=1
                correct+=(lam*(pre_x==x_t).sum().item()+(1-lam)*(pre_y==y_t).sum().item())
                loss=lam * criterion(out,x_la) + (1 - lam) * criterion(out, y_la)
                fen_loss+=loss
            loss=fen_loss/len(zhege)
            tmp_loss+=loss.float()
            if all_batch % 18==0:
                tmp_loss=tmp_loss/18
                all_loss.append(tmp_loss)
                tmp_loss=0.
                x_axi+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_batch+=1
        #TODO:early stop
        print(epoch+1," accuracy:{}",correct/total)
        if (abs(tmp_num-correct/total)<0.005) | (tmp_num>correct/total):
            brea_num+=1
        else:
            brea_num=0
        if brea_num==2:
            print("!!!error!!!")
            break
        tmp_num=correct/total
    model_name='.\model'+str(model_num)+'.pth'
    torch.save(model,model_name)
    return model,x_axi,all_loss
    for x in locals().keys():
        del locals()[x]
    gc.collect()