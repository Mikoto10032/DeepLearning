# coding: UTF-8
#博客链接：https://blog.csdn.net/White_Idiot/article/details/78816850
#https://blog.csdn.net/m0_37870649/article/details/79465917
#下载IInception_V3模型，读取模型，保存模型的图结构
import tensorflow as tf
import os
import tarfile
import requests
import glob
import os.path
import random
import numpy as np
from tensorflow.python.platform import gfile
"""
# inception模型下载地址
inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# 模型存放地址
inception_pretrain_model_dir = "inception_model"
if not os.path.exists(inception_pretrain_model_dir):
    os.makedirs(inception_pretrain_model_dir)

# 获取文件名，以及文件路径
filename = inception_pretrain_model_url.split('/')[-1]
filepath = os.path.join(inception_pretrain_model_dir, filename)

# 下载模型
if not os.path.exists(filepath):
    print("download: ", filename)
    r = requests.get(inception_pretrain_model_url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print("finish:", filename)

# 解压文件
tarfile.open(filepath, 'r:gz').extractall(inception_pretrain_model_dir)
"""

# 模型结构存放文件
log_dir = 'D:/tmp/inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# classify_image_graph_def.pb为google训练好的模型
inception_graph_def_file = os.path.join('D:/tensorflow_models/inception-2015-12-05', 'classify_image_graph_def.pb')

with tf.Session() as sess:
    with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='Inception_V3')
    writer = tf.summary.FileWriter(log_dir, sess.graph) #保存图的结构
    #writer = tf.train.SummaryWriter(log_dir, sess.graph)
    writer.close()