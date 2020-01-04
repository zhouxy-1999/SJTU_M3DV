import numpy as np
import csv

mix_index = 30      #我们只选取三维坐标均为max_index到mix_index的3D体素,减少数据量
max_index = 70

def get_data(path):
    if path=="train_val":
        csv_file=csv.reader(open("train_val.csv",'r'))
    else:
        csv_file=csv.reader(open("test.csv",'r'))
    content=[]
    for line in csv_file:
        content.append(line)
    data_len=len(content)
    data_voxel=[]
    data_seg=[]
    label=[]

    for i in range(1,data_len):
        data_path=path+'/'+content[i][0]+".npz"
        tmp=np.load(data_path)
        y=[];h=[]
        seg=tmp['seg'];voxel=tmp['voxel']*(tmp['seg']*0.8+0.2)
        seg=seg.astype(int);voxel=voxel/255
        seg=seg.tolist()
        # 对seg进行处理
        for j in seg[mix_index:max_index]:
            x=[]
            for k in j[mix_index:max_index]:
                x.append(k[mix_index:max_index])
            y.append(x)
        # 对voxel进行处理
        for j in voxel[mix_index:max_index]:
            vox=[]
            for k in j[mix_index:max_index]:
                vox.append(k[mix_index:max_index])
            h.append(vox)
        data_seg.append(y);data_voxel.append(h)
        tmp_result=float(int(content[i][1])-int('0'))
        label.append(tmp_result)
    return data_seg,label,content,data_voxel
