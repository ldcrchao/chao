%% ������С��ʱƵͼ
clear;clc;close all;format compact
%% �����ź�
load AI2_data1280_process_8k.mat
N=1280;   %���ݳ���
fs=8000;    % ����Ƶ��
t=0:1/fs:N/fs;

%% С���任ʱƵͼ
wavename='cmor3-3';%cmor�Ǹ�MorletС��������3��3��ʾFb��Fc��Fb�Ǵ��������Fc��С������Ƶ�ʡ�
totalscal=256;
Fc=centfrq(wavename); % С��������Ƶ��
c=2*Fc*totalscal;
scals=c./(1:totalscal);
f=scal2frq(scals,wavename,1/fs); % ���߶�ת��ΪƵ��
%% ѵ����
[m,n]=size(train_X);
[~,label]=max(train_Y,[],2);
for i=1:m
    i
    coefs=cwt(train_X(i,:),scals,wavename); % ������С��ϵ��
    % img=abs(coefs)/max(max(abs(coefs)));
    imagesc(t,f,abs(coefs)/max(max(abs(coefs))));
    set(gca,'position',[0 0 1 1])
    fname=['С��ʱƵ8k/train_img/',num2str(i),'-',num2str(label(i)),'.jpg'];
    saveas(gcf,fname);
end
%% ��֤��
[m,n]=size(valid_X);
[~,label]=max(valid_Y,[],2);
for i=1:m
    i
    coefs=cwt(valid_X(i,:),scals,wavename); % ������С��ϵ��
    % img=abs(coefs)/max(max(abs(coefs)));
    imagesc(t,f,abs(coefs)/max(max(abs(coefs))));
    set(gca,'position',[0 0 1 1])
    fname=['С��ʱƵ8k/valid_img/',num2str(i),'-',num2str(label(i)),'.jpg'];
    saveas(gcf,fname);
end
%% ���Լ�
[m,n]=size(test_X);
[~,label]=max(test_Y,[],2);
for i=1:m
    i
    coefs=cwt(test_X(i,:),scals,wavename); % ������С��ϵ��
    % img=abs(coefs)/max(max(abs(coefs)));
    imagesc(t,f,abs(coefs)/max(max(abs(coefs))));
    set(gca,'position',[0 0 1 1])
    fname=['С��ʱƵ8k/test_img/',num2str(i),'-',num2str(label(i)),'.jpg'];
    saveas(gcf,fname);
end