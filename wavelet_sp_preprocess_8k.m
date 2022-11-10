%% 批处理小波时频图
clear;clc;close all;format compact
%% 加载信号
load AI2_data1280_process_8k.mat
N=1280;   %数据长度
fs=8000;    % 采样频率
t=0:1/fs:N/fs;

%% 小波变换时频图
wavename='cmor3-3';%cmor是复Morlet小波，其中3－3表示Fb－Fc，Fb是带宽参数，Fc是小波中心频率。
totalscal=256;
Fc=centfrq(wavename); % 小波的中心频率
c=2*Fc*totalscal;
scals=c./(1:totalscal);
f=scal2frq(scals,wavename,1/fs); % 将尺度转换为频率
%% 训练集
[m,n]=size(train_X);
[~,label]=max(train_Y,[],2);
for i=1:m
    i
    coefs=cwt(train_X(i,:),scals,wavename); % 求连续小波系数
    % img=abs(coefs)/max(max(abs(coefs)));
    imagesc(t,f,abs(coefs)/max(max(abs(coefs))));
    set(gca,'position',[0 0 1 1])
    fname=['小波时频8k/train_img/',num2str(i),'-',num2str(label(i)),'.jpg'];
    saveas(gcf,fname);
end
%% 验证集
[m,n]=size(valid_X);
[~,label]=max(valid_Y,[],2);
for i=1:m
    i
    coefs=cwt(valid_X(i,:),scals,wavename); % 求连续小波系数
    % img=abs(coefs)/max(max(abs(coefs)));
    imagesc(t,f,abs(coefs)/max(max(abs(coefs))));
    set(gca,'position',[0 0 1 1])
    fname=['小波时频8k/valid_img/',num2str(i),'-',num2str(label(i)),'.jpg'];
    saveas(gcf,fname);
end
%% 测试集
[m,n]=size(test_X);
[~,label]=max(test_Y,[],2);
for i=1:m
    i
    coefs=cwt(test_X(i,:),scals,wavename); % 求连续小波系数
    % img=abs(coefs)/max(max(abs(coefs)));
    imagesc(t,f,abs(coefs)/max(max(abs(coefs))));
    set(gca,'position',[0 0 1 1])
    fname=['小波时频8k/test_img/',num2str(i),'-',num2str(label(i)),'.jpg'];
    saveas(gcf,fname);
end