
# coding: utf-8
# 小波时频图+卷积神经网络
# In[1]: 导入必要的库函数


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from utils import read_directory
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# In[2] 加载数据
num_classes=10
height=64
width=64
x_train,y_train=read_directory('小波时频8K/train_img',height,width,normal=1)
x_valid,y_valid=read_directory('小波时频8K/valid_img',height,width,normal=1)
# 转换为torch的输入格式
train_features = torch.tensor(x_train).type(torch.FloatTensor)
train_labels = torch.tensor(y_train).type(torch.LongTensor)
valid_features = torch.tensor(x_valid).type(torch.FloatTensor)
valid_labels = torch.tensor(y_valid).type(torch.LongTensor)

# In[3]: 参数设置
learning_rate = 0.01#学习率
num_epochs = 80#迭代次数
batch_size = 20 #batchsize  24最佳

train_data=TensorDataset(train_features, train_labels)
valid_data=TensorDataset(valid_features, valid_labels)

train_loader = DataLoader(train_data, batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size, shuffle=False)


# In[4]:
# 模型设置


class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        # 输入为64*64*3 的图片
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),    #卷积，64-5+1=60 ->60*60*6
            nn.MaxPool2d(kernel_size=2),       #池化，60/2=30 ->  30*30*6
            nn.ReLU(),                         #relu激活
            nn.Conv2d(6, 16, kernel_size=5),   #卷积，30-5+1=26 ->26*26*16
            nn.MaxPool2d(kernel_size=2),       #池化，26/2=13 ->13*13*16
            nn.ReLU()                          #relu激活
        )

        self.classifier = nn.Sequential(
            nn.Linear(2704, 120),              #全连接 120
            nn.ReLU(),                         #relu激活
            nn.Dropout(0.5),                   #dropout 0.5
            nn.Linear(120, 84),                #全连接 84
            nn.ReLU(),                         #relu激活
            nn.Linear(84, num_classes),        #输出 5
        )


    def forward(self, x):
        x = self.features(x)#进行卷积+池化操作提取图片特征
        logits = self.classifier(x.view(-1, 2704))#将上述特征拉伸为向量输入进全连接层实现分类
        probas = F.softmax(logits, dim=1)# softmax分类器
        return logits, probas


# 计算准确率与loss，就是把所有数据分批计算准确率与loss
def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    l=0
    for features, targets in data_loader:

        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        cost = loss(logits, targets)
        _, predicted_labels = torch.max(probas, 1)
        
        num_examples += targets.size(0)
       # print(num_examples)
       # print("/n")
        l += cost.item()
        correct_pred += (predicted_labels == targets).sum()
       # print(correct_pred)
    # print(l/num_examples,"      ",correct_pred.float()/num_examples * 100)
    return l/num_examples,correct_pred.float()/num_examples * 100,predicted_labels,targets

    
model = ConvNet(num_classes=num_classes)# 加载模型
model = model.to(device)#传进device
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)#sgd优化器
loss=torch.nn.CrossEntropyLoss()#交叉熵损失

# In[5]:

    
train_loss,valid_loss=[],[]
train_acc,valid_acc=[],[]
for epoch in range(num_epochs):
    model = model.train()#启用dropout
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        cost = loss(logits, targets)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    
    model = model.eval()#关闭dropout
    with torch.set_grad_enabled(False): # save memory during inference
        trl,trac,_,_=compute_accuracy(model, train_loader)
        val,vaac,_,_=compute_accuracy(model, valid_loader)
        print('Epoch: %03d/%03d training accuracy: %.2f%% validing accuracy: %.2f%%' %(
              epoch+1, num_epochs, 
              trac,
              vaac)
              )

    train_loss.append(trl)
    valid_loss.append(val)
    
    train_acc.append(trac)
    valid_acc.append(vaac)
    
torch.save(model,'model/W_CNN.pkl')#保存整个网络参数

# In[] 
#loss曲线
plt.figure()
plt.plot(np.array(train_loss),label='train')
plt.plot(np.array(valid_loss),label='valid')
plt.legend()
plt.show()
# accuracy 曲线
plt.figure()
plt.plot(np.array(train_acc),label='train')
plt.plot(np.array(valid_acc),label='valid')
plt.legend()
plt.show()
plt.savefig('curve.png', dpi=800, bbox_inches='tight')

# In[6]: 利用训练好的模型 对test_img图片进行分类

model=torch.load('model/W_CNN.pkl')#加载模型
#提取测试集图片
x_test,y_test=read_directory('小波时频8K/test_img',height,width,normal=1)#
test_features = torch.tensor(x_test).type(torch.FloatTensor)
test_labels = torch.tensor(y_test).type(torch.LongTensor)
test_data=TensorDataset(test_features, test_labels)
test_loader = DataLoader(test_data, batch_size, shuffle=False)

model = model.eval()
aaa,teacc,pred,target=compute_accuracy(model, test_loader)

##  return l / num_examples,  correct_pred.float() / num_examples * 100,  predicted_labels,  targets


#%%

print('测试集正确率为：',teacc.item(),'%')

print(pred)


