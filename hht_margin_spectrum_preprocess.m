%% ������HHT�߼���
clear;clc;close all;format compact;addpath('pack_hht')
%% �����ź�
load data_process.mat
N=864;
fs=48000;
t=0:1/fs:N/fs;

%% ѵ����
[m,n]=size(train_X);
[~,label]=max(train_Y,[],2);
tz=[];
for i=1:m
    
    x1=train_X(i,:);%����ź�
    u=emd(x1);
    [A,fa,tt]=hhspectrum(u);
    [E,tt1]=toimage(A,fa,tt,length(tt));%
    %E=flipud(E);%ʹ�������·�ת������õ��ı߼���Ƶ���Ǵ����ҵݼ�
    bjp=[];
    for k=1:size(E,1)
        bjp(k)=sum(E(k,:))*N/fs/2; 
    end
    PE= bjp/max(bjp);%��ÿ���ڵ�ĸ���,��һ��    
    tz(i,:)=PE;
end
x_train=tz;
y_train=label;
save HHT�߼���/train_data.mat x_train y_train
disp('ѵ�����������')
%% ��֤��
[m,n]=size(valid_X);
[~,label]=max(valid_Y,[],2);
tz=[];
for i=1:m
    x1=valid_X(i,:);%����ź�
        u=emd(x1);
    [A,fa,tt]=hhspectrum(u);
    [E,tt1]=toimage(A,fa,tt,length(tt));%
    %E=flipud(E);%ʹ�������·�ת������õ��ı߼���Ƶ���Ǵ����ҵݼ�
    bjp=[];
    for k=1:size(E,1)
        bjp(k)=sum(E(k,:))*N/fs/2; 
    end
    PE= bjp/max(bjp);%��ÿ���ڵ�ĸ���,��һ��    
    tz(i,:)=PE;
end
x_valid=tz;
y_valid=label;
save HHT�߼���/valid_data.mat x_valid y_valid
disp('��֤���������')

%% ���Լ�
[m,n]=size(test_X);
[~,label]=max(test_Y,[],2);
tz=[];
for i=1:m
    x1=test_X(i,:);%����ź�
       u=emd(x1);
    [A,fa,tt]=hhspectrum(u);
    [E,tt1]=toimage(A,fa,tt,length(tt));%
    %E=flipud(E);%ʹ�������·�ת������õ��ı߼���Ƶ���Ǵ����ҵݼ�
    bjp=[];
    for k=1:size(E,1)
        bjp(k)=sum(E(k,:))*N/fs/2; 
    end
    PE= bjp/max(bjp);%��ÿ���ڵ�ĸ���,��һ��    
    tz(i,:)=PE;
end
x_test=tz;
y_test=label;
save HHT�߼���/test_data.mat x_test y_test
disp('���Լ��������')
