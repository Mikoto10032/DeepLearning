# -*- coding: utf-8 -*-
# @Author: LiangHaozan
# @Date:   2018-04-10 14:28:13
# @Last Modified time: 2018-04-13 19:17:26

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

#数据参数
MODEL_DIR = 'model/'  # inception-v3模型的文件夹
MODEL_FILE = 'classify_image_graph_def.pb'  # inception-v3模型文件名
CACHE_DIR = 'data/tmp/bottleneck'  # 图像的特征向量保存地址
INPUT_DATA = 'data/flower_photos'  # 图片数据文件夹
VALIDATION_PERCENTAGE = 10  # 验证数据的百分比
TEST_PERCENTAGE = 10  # 测试数据的百分比

#Inception_V3参数
#BOTTLENECK：瓶颈层
BOTTLENECK_TENSOR_SIZE=2048 # inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称，通过这个名称可以从导入的模型得到对应的变量，然后通过sess.run({变量：数据})得到值
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

#神经网络训练参数
LEARNING_RATE=0.01  #学习率
STEPS=100   #训练步数
BATCH=100   #训练样本批量大小
CHECKPOINT_EVERY=100    #每固定步数检测点
NUM_CHECKPOINTS=5   #检查点数

#从data文件夹中读取所有图片按照训练、验证、测试分开
#@Param validation_percentage,验证集比例；test_percentage,测试集比例
#@return result，结果字典，key是图像类别，value是字典
def create_image_lists(validation_percentage,test_percentage):
    "先创建字典保存样本信息，读取所有子目录"
    result={}   #结果字典，用来保存所有图像信息，key为类别名称。value也是字典，存储了所有的图片名称
    sub_dirs=[x[0] for x in os.walk(INPUT_DATA)]    #获取所有子目录
    is_root_dir=True #第一个目录为当前目录，需要忽略

    #分别对每个子目录进行操作
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir=False
            continue

        #获取当前目录所有有效图片
        extensions={'jpg','jpeg','JPG','JPEG'}
        file_list=[]    #存储所有图像路径
        dir_name=os.path.basename(sub_dir)  #获取路径的最后一个目录名字
        for  extension in extensions:
            file_glob=os.path.join(INPUT_DATA,dir_name,'*.'+extension)  #构造匹配所有图片的表达式
            file_list.extend(glob.glob(file_glob))  #glob.glob(reg),找到匹配reg的所有文件路径；file_list.extend（），append路径到file_list里
        if not file_list:
            continue

        #将当前类别的图片随机分成训练数据集、测试数据集、验证数据集
        label_name=dir_name.lower() #将字符串全部转化为小写字母
        training_images=[]
        testing_images=[]
        validation_images=[]
        for file_name in file_list:
            base_name=os.path.basename(file_name)   #获取图片名称
            chance=np.random.randint(100)   #随机产生100个数代表百分比,下面按照概率分成训练集、测试集、验证集并且保存图片名称
            if chance<validation_percentage:
                validation_images.append(base_name)
            elif chance<validation_percentage+test_percentage:
                testing_images.append(base_name)
            else :
                training_images.append(base_name)

        #将当前类别的数据集放进结果字典;大概是 {'label0':{'dir':xxx,'training':xxx,'testing':xxx,'validation':xxx}}
        result[label_name]={
            'dir':dir_name,
            'training':training_images,
            'testing':testing_images,
            'validation':validation_images
        }

    #返回整理好的数据
    return result

#通过类别、所属数据集、图片编号获取一张图片的地址
#@Param image_lists,create_image_lists返回的result；image_dir，数据集路径；label_name,图片类别；
#index，图片下标；category，数据集类型（test,validation,train）
#@return 图片绝对路径
def get_image_path(image_lists,image_dir,label_name,index,category):
    label_lists=image_lists[label_name] #获取给定类别的所有图片
    category_lists=label_lists[category]    #根据给出数据集类型获取该集合所有图片
    mod_index=index%len(category_lists) #规范图片的索引
    base_name=category_lists[mod_index] #获取图片的文件名(xxx.jpg)
    sub_dir=label_lists['dir']  #获取当前类别的目录名
    full_path=os.path.join(image_dir,sub_dir,base_name) #图片的绝对路径
    return full_path

# 通过类别名称、所属数据集、图片编号获取特征向量值的地址
# @Param image_lists,按类别存放所有图片信息的字典；label_name,图片类别；index，图片下标；category，数据集类型（test,validation,train）
# @return 图片特征向量值的地址
def get_bottleneck_path(image_lists,label_name,index,category):
    return get_image_path(image_lists,CACHE_DIR,label_name,index,category)+'.txt'      #data/tmp/bottleneck/xxx.jpg.txt

#使用Inception_V3模型处理图片获得特征向量
#@Param image_data，读进来的图片矩阵；image_data_tensor,图像输入张量对应的名称;bottleneck_tensor,inception-v3模型中代表瓶颈层结果的张量名称
#@return 图片在瓶颈层的特征向量
def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
    bottleneck_values=sess.run(bottleneck_tensor,{image_data_tensor:image_data})    #大概是得到加载模型后训练的得到fc层的张量
    bottleneck_values=np.squeeze(bottleneck_values) #将四维数组压成一维数组
    return bottleneck_values

#获取一张图片经过Inception-V3模型处理后的特征向量
#@Param image_lists,按类别存放所有图片信息的字典；label_name,图片类别；index，图片下标；category，数据集类型（test,validation,train）
#image_data_tensor,图像输入张量对应的名称;bottleneck_tensor,inception-v3模型中代表瓶颈层结果的张量名称
#@return 图片在瓶颈层的特征向量
def get_or_ctreate_bottleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor):
    #获取一张图片对应的特征向量的路径
    label_lists=image_lists[label_name]
    sub_dir=label_lists['dir']
    sub_dir_path=os.path.join(CACHE_DIR,sub_dir)
    if not os.path.exists(sub_dir_path):    #如果不存在该类别的特征向量目录，则创建一个目录
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          category)

    #如果该特征向量文件不存在，则通过Inception-V3模型获取
    if not os.path.exists(bottleneck_path):
        image_path=get_image_path(image_lists,INPUT_DATA,label_name,index,category)  #获取图片绝对路径
        image_data=gfile.FastGFile(image_path,'rb').read()  #获取图片内容
        bottleneck_values=run_bottleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)   #通过Inception_V3计算特征向量

        #将特征向量存入文件
        bottleneck_string=','.join(str(x) for x in bottleneck_values)   #str.join(x),用str连接x
        with open(bottleneck_path,'w')  as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    else :
        #否则直接从文件中获取特征向量
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string=bottleneck_file.read()    #x,x,x,x
        bottleneck_values=[float(x) for x in bottleneck_string.split(',')]  #读取文件中保存的字符串，切割后转化为float

    return bottleneck_values    #返回得到的特征向量

#随机获取一个batch图片作为训练数据
#@Param n_classes,类别数目；image_lists，包含各类图片的字典；how_many，batch大小；
#category，样本类别；jpeg_data_tensor,bottleneck_tensor，分别对应模型中的tensor变量名称
#@return bottlenecks,一批图片对应的特征向量；ground_truths，一批图片对应的类别
def get_random_cached_bottlenecks(sess,n_classes,image_lists,how_many,category,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks=[]  #瓶颈层输出
    ground_truths=[]    #参考标准，可以理解为正确的标签
    for _ in range(how_many):
        # 随机一个类别和图片编号加入当前的训练数据
        label_index=random.randrange(n_classes) #类别下标;random.randrange(start,stop,steps),[start,stop）,步长为steps，从这个序列中随机选一个值
        label_name=list(image_lists.keys())[label_index]    #类别名称；list() 方法用于将元组转换为列表；字典(Dictionary) keys() 函数以列表返回一个字典所有的键
        image_index=random.randrange(65535) #随机一个下标
        bottleneck=get_or_ctreate_bottleneck(sess,image_lists,label_name,image_index,category,jpeg_data_tensor,bottleneck_tensor)   #得到该图片的特征向量
        ground_truth=np.zeros(n_classes,dtype=np.float32)   #[0,0,0,0,0]
        ground_truth[label_index]=1.0   #例如[0,1.0,0,0,0],其中1对应正确的类别

        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks,ground_truths

#获取全部测试数据
#@Param n_classes,类别数目；image_lists，包含各类图片的字典；image_data_tensor,图像输入张量对应的名称;
#bottleneck_tensor,inception-v3模型中代表瓶颈层结果的张量名称
#@return
def get_test_bottelenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks=[]  #瓶颈层输出
    ground_truths=[]    #参考标准，可以理解为正确的标签
    label_name_list=list(image_lists.keys())    #类别列表
    #枚举所有类别和每个类别中的测试图片
    for label_index,label_name in enumerate(label_name_list):   # 0,label1;1,label2......
        category='testing'
        for index,unused_base_name in enumerate(image_lists[label_name][category]): #0,xxx.jpg
            bottleneck=get_or_ctreate_bottleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor)
            ground_truth=np.zeros(n_classes,dtype=np.float32)
            ground_truth[label_index]=1.0

            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)

    return bottlenecks,ground_truths

def main(_):
    #读取所有图片
    image_lists=create_image_lists(VALIDATION_PERCENTAGE,TEST_PERCENTAGE)   #image_lists,按类别存放所有图片信息的字典
    n_classes=len(image_lists.keys())   #类别个数

    #定义计算图，导入训练好的模型，然后添加自己的模型，也就是Softmax分类器
    with tf.Graph().as_default() as graph:
        #读取训练好的Inception_V3模型,run_bottleneck_on_image()函数才能够run出特征向量
        with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(f.read())
            # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量，然后run就能得到值
            bottleneck_tensor,jpeg_data_tensor=tf.import_graph_def(
                graph_def,return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME]) #瓶颈层输出

        #定义新的神经网络输入
        bottleneck_input=tf.placeholder(tf.float32,[None,BOTTLENECK_TENSOR_SIZE],name='BottleneckInputPlaceholder') #[-1,2048]
        # 定义新的标准答案输入
        ground_truth_input=tf.placeholder(tf.float32,[None,n_classes],name='GroundTruthInput')
        # 定义一层全连接层解决新的图片分类问题
        with tf.name_scope('final_training_ops'):
            weights=tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE,n_classes],stddev=0.1)) #全连接层权重
            bias=tf.Variable(tf.zeros([n_classes]))    #偏置
            logits=tf.matmul(bottleneck_input,weights)+bias     #未归一化概率
            final_tensor=tf.nn.softmax(logits)  #最后的分类

        #定义交叉熵损失函数
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,labels=ground_truth_input)    #交叉熵矩阵，[-1,4]
        cross_entropy_mean=tf.reduce_mean(cross_entropy) #[1]

        train_step=tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)    #梯度下降最小化交叉熵

        #计算正确率
        with tf.name_scope('evaluation'):
            correct_prediction=tf.equal(tf.argmax(final_tensor,1),tf.argmax(ground_truth_input,1))  #预测矩阵，形如[[True] [True] [False]]
            evaluation_step=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  #正确率

    #训练过程
    with tf.Session(graph=graph) as sess:
        init=tf.global_variables_initializer().run()

        #模型和摘要的保存目录
        import time
        timestamp=str(int(time.time()))
        out_dir=os.path.abspath(os.path.join(os.path.curdir,'runs',timestamp))   #摘要保存目录
        print('\nWriting to {}\n',format(out_dir))

        #损失值和正确率摘要
        loss_summary=tf.summary.scalar('loss',cross_entropy_mean)
        acc_summary=tf.summary.scalar('accuracy',evaluation_step)

        #训练摘要(训练时)
        train_summary_op=tf.summary.merge([loss_summary,acc_summary])   #合并摘要
        train_summary_dir=os.path.join(out_dir, 'summaries', 'train')   #训练摘要保存目录
        train_summary_writer=tf.summary.FileWriter(train_summary_dir,sess.graph)    #定义writer

        #开发摘要(验证时)
        dev_summary_op=tf.summary.merge([loss_summary,acc_summary])
        dev_summary_dir=os.path.join(out_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)     #定义writer

        #保存检查点(持久化训练好的模型)
        checkpoint_dir=os.path.abspath(os.path.join(out_dir, 'checkpoints'))    #checkpoint目录
        checkpoint_prefix=os.path.abspath(os.path.join(checkpoint_dir,'model'))  #checkpoint前缀
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            saver=tf.train.Saver(tf.global_variables(),max_to_keep=NUM_CHECKPOINTS) #创建saver类来保存模型
        #开始每一步训练
        for i in range(STEPS+1):
            #每次选取一个batch的训练数据
            train_bottlenecks,train_ground_truths=get_random_cached_bottlenecks(
                sess,n_classes,image_lists,BATCH,'training',jpeg_data_tensor,bottleneck_tensor) #category='training'的特征向量、正确标签
            #开始训练新模型
            _,train_summaries=sess.run([train_step,train_summary_op],feed_dict={
                bottleneck_input:train_bottlenecks,ground_truth_input:train_ground_truths
            })   #feed_dict{新模型的输入层：训练集的瓶颈层向量，新模型的正确输出标签：训练集的标签

            #保存每步的摘要
            train_summary_writer.add_summary(train_summaries,i) #训练时注释掉，节省时间和磁盘空间

            #在验证集上测试正确率
            if i%50 == 0 or i+1==STEPS:
                validation_bottlenecks,validation_ground_truths=get_random_cached_bottlenecks(
                    sess,n_classes,image_lists,BATCH,'validation',jpeg_data_tensor,bottleneck_tensor) #category='validation'的特征向量、正确标签
                validation_accuracy,dev_summaries=sess.run(
                    [evaluation_step,dev_summary_op],feed_dict={
                    bottleneck_input:validation_bottlenecks,ground_truth_input:validation_ground_truths
                    })  #feed_dict{新模型的输入层：验证集的瓶颈层向量，新模型的正确输出标签：验证集的标签
                print(
                    'step %d:Validation accuracy on random sampled %d examples=%.1f%%'
                    %(i,BATCH,validation_accuracy*100))

            # 每隔checkpoint_every保存一次模型和测试摘要
            if i%CHECKPOINT_EVERY==0 :
                dev_summary_writer.add_summary(dev_summaries,i) #训练时注释掉，节省时间和磁盘空间
                path=saver.save(sess,checkpoint_prefix,global_step=i)
                print('Saved model checkpoint to {}\n'.format(path))

        #最后在测试集上测试正确率
        test_bottelenecks,test_ground_truths=get_test_bottelenecks(
            sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor)  #category='testing'的特征向量、正确标签
        test_accuracy=sess.run([evaluation_step],feed_dict={
            bottleneck_input:test_bottelenecks,ground_truth_input:test_ground_truths
            })  #feed_dict{新模型的输入层：测试集的瓶颈层向量，新模型的正确输出标签：测试集的标签
        #print(type(test_accuracy[0]))
        #print(test_accuracy[0]*100)
        print('Final test accuracy = %.1f%%' % (test_accuracy[0]*100))

        # 保存标签
        output_labels = os.path.join(out_dir, 'labels.txt')
        with tf.gfile.FastGFile(output_labels, 'w') as f:
            keys = list(image_lists.keys())
            for i in range(len(keys)):
                keys[i] = '%2d -> %s' % (i, keys[i])
            f.write('\n'.join(keys) + '\n')

if __name__ == '__main__':
    tf.app.run()