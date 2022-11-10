# In[1] 加载必要的库
import numpy as np
import os
from PIL import Image
from torchvision import transforms as tfs
import skimage as io

# In[2] 定义数据提取函数

def read_directory(directory_name,height,width,normal):
    #directory_name='train_img'
    #height=64
    #width=64
    #normal=1
    file_list=os.listdir(directory_name)
    img = []
    label0=[]
    
    for each_file in file_list:
        img0 = Image.open(directory_name + '/'+each_file)
        gray = img0.resize((height,width))
        img.append(np.array(gray).astype(np.float))
        label0.append(float(each_file.split('.')[0][-1]))
    if normal:
        data = np.array(img)/255.0#归一化
    else:
        data = np.array(img)
    data=data.reshape(-1,3,height,width)
    label=np.array(label0)
    return data,label 
