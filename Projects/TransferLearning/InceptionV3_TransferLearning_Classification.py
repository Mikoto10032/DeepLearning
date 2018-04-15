# -*- coding: utf-8 -*-
# @Author: Lianghaozan
# @Date:   2018-04-13 17:00:44
# @Last Modified time: 2018-04-13 19:26:46

import tensorflow as tf
import numpy as np

#模型目录
CHECKPOINT_DIR = 'runs/1523618446/checkpoints'
INCEPTION_MODEL_FILE = 'model/classify_image_graph_def.pb'
#BOTTLENECK_FILE = 'test_bottleneck'

# inception-v3模型参数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

# 测试数据
file_path = 'test_image/394990940_7af082cf8d_n.jpg'
y_test = [2]    #对应标签

#读取数据
image_data=tf.gfile.FastGFile(file_path,'rb').read()

#评估
checkpoint_file=tf.train.latest_checkpoint(CHECKPOINT_DIR)

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

        #加载元图和变量
        saver=tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess,checkpoint_file)

        #通过名称从图中获得输入占位符
        input_x=graph.get_operation_by_name('BottleneckInputPlaceholder').outputs[0]

        #通过名字从图中获取预测标签下标
        predictions=graph.get_operation_by_name('evaluation/ArgMax').outputs[0]

        # 收集预测值
        all_predictions=[]
        all_predictions=sess.run(predictions, {input_x: bottleneck_values})
        print(all_predictions)