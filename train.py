# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:16:54 2019

@author: xinxi ss
"""


#%% 先加载进必要的文件
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到
import matplotlib.pyplot as plt
import numpy as np
import input_Data
#import model_new2
import model_Unet_1
tf.reset_default_graph()

#%%输入各种文件的路径 
train_images_path = 'H:\\tzw\\train_imgs_auto\\'
val_images_path = 'H:\\tzw\\val_imgs_auto\\'

train_labels_path = '.\\data\\train_labels_128\\'
val_labels_path = '.\\data\\val_labels_128\\'

train_TFRecord_path = '.\\tfrecord_4000\\train.tfrecord'
val_TFRecord_path = '.\\tfrecord_4000\\val.tfrecord'
model_save_path = '.\\model_mnist_2\\model_mnist_2.ckpt'

# file = '\\11001.png'
result_save_path = 'G:\\tzw\\dataset\\result\\mnist_4000\\pre_training\\'
# result_save_path = '.\\result\\DIP_TL\\'

step_num = 30000     # 训练步数
img_W = 128          # 图像宽度
img_H = 128          # 图像高度
lab_W = 128          #标签宽度
lab_H = 128          #标签高度
batch_size = 5       # 每个mini-batch含有的样本数量 
keep_prob = 1        #dropout的概率


#%%读取相应的文件
# input_Data.generate_TFRecordfile(train_images_path,train_labels_path,train_TFRecord_path)   #调用generate_TFRecordfile函数生成TFRecord文件记录训练数据
# input_Data.generate_TFRecordfile(val_images_path,val_labels_path,val_TFRecord_path)         #调用generate_TFRecordfile函数生成TFRecord文件记录测试数据

#根据TFRecord文件中的内容生成符合mini-batch方法训练要求的 Image_Batch,Label_Batch
Train_Images_Batch,Train_Labels_Batch = input_Data.get_batch(train_TFRecord_path,img_W,img_H,lab_W,lab_H,batch_size)
Test_Images_Batch,Test_Labels_Batch = input_Data.get_batch(val_TFRecord_path,img_W,img_H,lab_W,lab_H,batch_size)

#%% 编写训练代码程序
def train(Train_Images_Batch,Train_Labels_Batch,Test_Images_Batch,Test_Labels_Batch):
    
    with tf.variable_scope('input'):
        x = tf.placeholder(tf.float32, shape=[batch_size,img_W,img_H,1],name = 'images')
        y_label = tf.placeholder(tf.float32,shape=[batch_size,lab_W,lab_H,1],name = 'labels')
        keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')       #设置droupout的控制参数
        is_training = tf.placeholder(tf.bool,name = 'is_train')                           #设置BN的控制参数
        #tf.summary.image('images',x,4)
        #tf.summary.image('labels',y_label,4)
        rng = np.random.RandomState(824)
        inpt = tf.constant(rng.uniform(0, 0.1, size=(batch_size,img_W,img_H,1)), dtype = tf.float32) #输入是均匀随机分布的固定值
    
    y_conv = model_Unet_1.inference(x, img_W, img_H, batch_size, is_training)
    #tf.summary.image('outputs',y_conv,4)
    
    with tf.variable_scope('loss_function'):
        loss = tf.reduce_mean(tf.square(y_conv - y_label)) 
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.variable_scope('train_step'):
            train_op = optimizer.minimize(loss)
    
#    with tf.variable_scope('train_step'):
#        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    init_op = (tf.local_variables_initializer(),tf.global_variables_initializer())
    
    #merged = tf.summary.merge_all() # 整理所有的日志生成操作
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # 神经网络训练准备工作。变量初始化，线程启动
        sess.run(init_op)
        coord = tf.train.Coordinator() # 线程终止(should_stop=True,request_stop)
        threads = tf.train.start_queue_runners(sess=sess,coord=coord) # 启动多个线程操作同一队列
        
        # 初始化写日志的writer 将当前计算图写入日志
        #summary_writer = tf.summary.FileWriter(TensorBoard_path,tf.get_default_graph()) 

        try:
            for step in range(step_num):
                if coord.should_stop():
                    print('hello world')                
                    break
                train_images_batch,train_labels_batch = sess.run([Train_Images_Batch,Train_Labels_Batch])

                train_images_batch = np.reshape(train_images_batch,[batch_size,img_W,img_H,1]) 
                
                train_labels_batch = np.reshape(train_labels_batch,[batch_size,lab_W,lab_H,1])
       
                sess.run(train_op,feed_dict={x:train_images_batch,y_label:train_labels_batch,keep_prob:1.0,is_training:True})  # 将mini-batch feed给train_op 训练网络
                
                if  step%300 == 0:
                    test_images_batch,test_labels_batch = sess.run([Test_Images_Batch,Test_Labels_Batch])
                    test_images_batch = np.reshape(test_images_batch,[batch_size,img_W,img_H,1]) 
                    test_labels_batch = np.reshape(test_labels_batch,[batch_size,lab_W,lab_H,1])

                    train_loss = sess.run(loss,feed_dict={x:train_images_batch,y_label:train_labels_batch,keep_prob:1.0,is_training:True})
                    test_loss = sess.run(loss,feed_dict={x:test_images_batch,y_label:test_labels_batch,keep_prob:1.0,is_training:True})
                    
                    saver.save(sess, model_save_path)
                    print('[step %d]: loss on training set batch:%f  loss on testing set batch:%f' % (step,train_loss,test_loss))

                if step%500 == 0:
                    y_pred = sess.run(y_conv,feed_dict={x:test_images_batch,is_training:True,keep_prob:1.0})  # 将mini-batch feed给train_op 训练网络    
                    
                    y_real = test_labels_batch[0,:,:,:]
                    y_real = np.reshape(y_real,[lab_W,lab_H])
                    y_real = y_real*255
                    y_real = Image.fromarray(y_real.astype('uint8'))  #需要根据位深改变uint8或者uint16#
                    
                    img_i = test_images_batch[0,:,:,:]
                    img_i = np.reshape(img_i,[img_W,img_H])
                    #img_i = img_i*255
                    img_i = img_i*255
                    img_i = Image.fromarray(img_i.astype('uint16'))                    
        
                    img_o = y_pred[0,:,:,:]
                    img_o = np.reshape(img_o,[lab_W,lab_H])
                    img_o = img_o*255
                    img_o[img_o>255] = 255                    
                    img_o = Image.fromarray(img_o.astype('uint8')) 
                    img_o.save(result_save_path+'re_step%d_img.bmp'%(step))
                    
                    
                    plt.subplot(131)
                    plt.imshow(y_real)
                    plt.subplot(132)
                    plt.imshow(img_i)
                    plt.show()
                    plt.subplot(133)
                    plt.imshow(img_o)
                    plt.show()                    

            y_pred = sess.run(y_conv,feed_dict={x:test_images_batch,keep_prob:1.0,is_training:False})  # 将mini-batch feed给train_op 训练网络
            temp = y_pred[0,:,:,:]
            temp[temp<0]=0
            img_o = temp
            img_o = np.reshape(img_o,[lab_W,lab_H])
            img_o = Image.fromarray(img_o.astype('uint8'))                   
            plt.imshow(img_o)
            plt.show()

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
            coord.request_stop()
        finally:
            coord.request_stop()
            coord.join(threads)
        #summary_writer.close()
        saver.save(sess, model_save_path)
    
def main(argv=None):
    train(Train_Images_Batch,Train_Labels_Batch,Test_Images_Batch,Test_Labels_Batch)

if __name__=='__main__':
    main()





