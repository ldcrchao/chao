%% 批处理HHT边际谱
clear;clc;close all;format compact;addpath('pack_hht')
%% 加载信号
load data_process.mat
N=864;
fs=48000;
t=0:1/fs:N/fs;

%% 训练集
[m,n]=size(train_X);
[~,label]=max(train_Y,[],2);
tz=[];
for i=1:m
    
    x1=train_X(i,:);%轴承信号
    u=emd(x1);
    [A,fa,tt]=hhspectrum(u);
    [E,tt1]=toimage(A,fa,tt,length(tt));%
    %E=flipud(E);%使矩阵上下翻转，否则得到的边际谱频率是从左到右递减
    bjp=[];
    for k=1:size(E,1)
        bjp(k)=sum(E(k,:))*N/fs/2; 
    end
    PE= bjp/max(bjp);%求每个节点的概率,归一化    
    tz(i,:)=PE;
end
x_train=tz;
y_train=label;
save HHT边际谱/train_data.mat x_train y_train
disp('训练集处理完毕')
%% 验证集
[m,n]=size(valid_X);
[~,label]=max(valid_Y,[],2);
tz=[];
for i=1:m
    x1=valid_X(i,:);%轴承信号
        u=emd(x1);
    [A,fa,tt]=hhspectrum(u);
    [E,tt1]=toimage(A,fa,tt,length(tt));%
    %E=flipud(E);%使矩阵上下翻转，否则得到的边际谱频率是从左到右递减
    bjp=[];
    for k=1:size(E,1)
        bjp(k)=sum(E(k,:))*N/fs/2; 
    end
    PE= bjp/max(bjp);%求每个节点的概率,归一化    
    tz(i,:)=PE;
end
x_valid=tz;
y_valid=label;
save HHT边际谱/valid_data.mat x_valid y_valid
disp('验证集处理完毕')

%% 测试集
[m,n]=size(test_X);
[~,label]=max(test_Y,[],2);
tz=[];
for i=1:m
    x1=test_X(i,:);%轴承信号
       u=emd(x1);
    [A,fa,tt]=hhspectrum(u);
    [E,tt1]=toimage(A,fa,tt,length(tt));%
    %E=flipud(E);%使矩阵上下翻转，否则得到的边际谱频率是从左到右递减
    bjp=[];
    for k=1:size(E,1)
        bjp(k)=sum(E(k,:))*N/fs/2; 
    end
    PE= bjp/max(bjp);%求每个节点的概率,归一化    
    tz(i,:)=PE;
end
x_test=tz;
y_test=label;
save HHT边际谱/test_data.mat x_test y_test
disp('测试集处理完毕')
