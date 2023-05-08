# -*- coding: utf-8 -*-
"""
Created on May  5  2022



"""
@author: LEE

"""

1.用DeweSoft软件裁剪原始信号（选取特征明显的channel），每段10s左右，大约160k个点位，输出时选择采样频率在8K左右的mat格式(需配合Matlab2018b版本);
2.N种故障模式对应N个信号片段，放在OHP文件夹内；
3.用“原始数据图提取.PY”代码，将N个片段汇总成“data1280_process_8K.mat”；
4.用“wavelet_sp_preprocess_8K.m”代码，生成N种对应的小波时频图；
5.用“Main.PY”代码，读取图片并进行分类。

"""
