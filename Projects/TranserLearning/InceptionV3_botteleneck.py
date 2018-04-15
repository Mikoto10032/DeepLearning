# -*- coding: utf-8 -*-
# @Author: LiangHaozan
# @Date:   2018-04-12 10:21:50
# @Last Modified by:   Marte
# @Last Modified time: 2018-04-12 14:04:34

import tensorflow as tf
import numpy as np

#模型目录
INCEPTION_MODEL_FILE = 'model/classify_image_graph_def.pb'
#BOTTLENECK_FILE = 'test_bottleneck'

# inception-v3模型参数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

# 测试数据
file_path = 'test_image/394990940_7af082cf8d_n.jpg'
y_test = [4]

#读取数据
image_data=tf.gfile.FastGFile(file_path,'rb').read()

with tf.Graph().as_default() as graph:
    with tf.Session().as_default() as sess:
        #读取训练好的Inception_V3模型，得到一个图
        with tf.gfile.FastGFile(INCEPTION_MODEL_FILE,'rb') as f:    #打开模型
            graph_def=tf.GraphDef() #定义一个图
            graph_def.ParseFromString(f.read()) #将模型加载进图

        # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量
        bottleneck_tensor,jpeg_data_tensor=tf.import_graph_def(graph_def,return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])

        # 使用inception-v3处理图片获取特征向量
        bottleneck_values=sess.run(bottleneck_tensor,feed_dict={jpeg_data_tensor:image_data})
        print(bottleneck_values.shape)
        # 将[1,2048]维数组压缩成一维数组，由于全连接层输入时有batch的维度，所以用列表作为输入
        bottleneck_values=[np.squeeze(bottleneck_values)] #[[1 2 3]]

        bottleneck_string=','.join(str(x) for x in bottleneck_values[0])   #用,连接向量的每个数字成字符串
        with open(file_path+'.txt','w') as file:    #保存在txt
            file.write(bottleneck_string)
        print(type(bottleneck_values),bottleneck_values[0][0])
        #for x in bottleneck_values[0]:
        #    print(x)

        with open(file_path+'.txt','r') as file:    #读取txt
            bottleneck_string=file.read()
            bottleneck_values=[float(x) for x in bottleneck_string.split(',')]  #剪切出数值
        print(type(bottleneck_values),bottleneck_values)