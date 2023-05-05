# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:16:54 2019

@author: xinxi ss
"""


#%% 先加载进必要的文件
import tensorflow as tf
from PIL import Image  
import matplotlib.pyplot as plt
import numpy as np
import model_Unet_1
import Myfftshift

tf.reset_default_graph()

#%%输入各种文件的路径 
model_save_path = '.\\model_mnist_2\\model_mnist_2.ckpt'
#
# file = '.\\data\\val_imgs\\4360.png'
# ac_save_path = '.\\result\\DL\\' + file 
ac_save_path = 'H:\\tzw\\val_imgs_auto\\4360.png'
result_save_path = '.\\result\\mnist_4000\\fine_tune_4360_with_110\\'
label_path = '.\\data\\val_labels_128\\4360.jpg'
auto_result_save_path = '.\\result\\mnist_4000\\fine_tune_4360_auto_with_110\\'
#ac_save_path = '.\\data\\nobacknoise_auto\\test1\\2.png' 
#ac_save_path = '.\\data\\data_noise\\face_no_noise.png'
#resut_save_path = '.\\result\\DIP_TL\\data_noise\\'


dim = 128
img_W = 128          # 图像宽度
img_H = 128          # 图像高度
lab_W = 128          # 标签宽度
lab_H = 128          # 标签高度
batch_size = 1       # 每个mini-batch含有的样本数量 
keep_prob = 1        # dropout的概率
test_num = 1         # 待测试的样本数量
pad_size = 256       # 将所得结果填充0为这个尺寸

def Measure_step(img_input):
    img_input = tf.cast(img_input,dtype = tf.complex64)
    FF = tf.fft2d(img_input)
    f_mul = tf.multiply(tf.conj(FF),FF)
    IM = tf.ifft2d(f_mul)
    IM = Myfftshift.ifftshift(IM)
    IM = tf.abs(IM)    
    return IM


with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, shape=[batch_size,img_W,img_H,1],name = 'images')
    y_label = tf.placeholder(tf.float32,shape=[batch_size,lab_W,lab_H,1],name = 'labels')
    keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')                   #设置droupout的控制参数
    is_training = tf.placeholder(tf.bool,name = 'is_train')                     #设置BN的控制参数    
    #rand = tf.placeholder(shape=(1,img_W,img_H,1), dtype=tf.float32)
    rand = tf.placeholder(tf.float32,shape = (1,img_W,img_H,1))

#    rng = np.random.RandomState(824)
#    inpt = tf.constant(rng.uniform(0, 0.1, size=(1,img_W,img_H,1)), dtype = tf.float32) + rand #输入是均匀随机分布的固定值
      
y_conv = model_Unet_1.inference(x, img_W, img_H, batch_size, is_training) 

raw_measure = tf.reshape(x,[img_W,img_H])
raw_measure = (raw_measure - tf.reduce_min(raw_measure))/(tf.reduce_max(raw_measure) - tf.reduce_min(raw_measure))

#加支持域
out = tf.reshape(y_conv,[img_W,img_H])
mask = tf.ones([110,110],dtype=tf.float32)
mask = tf.pad(mask, [[int((dim - 110)/2),int((dim - 110)/2)],[int((dim - 110)/2),int((dim - 110)/2)]], mode='CONSTANT', name=None, constant_values=0)
# mask = plt.imread(path_4360)
# ss = tf.ones([128,128],dtype=tf.float32)
# ss1 = mask * ss
# ss2 = tf.zeros([128,128])

# out_pad = out * (ss - mask)

out_pad = out * mask
# out_pad = tf.pad(out, [[int((pad_size2 - dim)/2),int((pad_size2 - dim)/2)],[int((pad_size2 - dim)/2),int((pad_size2 - dim)/2)]], mode='CONSTANT', name=None, constant_values=0)
out_pad = tf.pad(out_pad, [[int((pad_size - dim)/2),int((pad_size - dim)/2)],[int((pad_size - dim)/2),int((pad_size - dim)/2)]], mode='CONSTANT', name=None, constant_values=0)
#对输出结果补零
# out = tf.reshape(y_conv,[img_W,img_H])
# out_pad = tf.pad(out, [[int((pad_size - dim)/2),int((pad_size - dim)/2)],[int((pad_size - dim)/2),int((pad_size - dim)/2)]], mode='CONSTANT', name=None, constant_values=0)
out_measure = Measure_step(out_pad)                                                           # 自相关
out_measure = tf.reshape(out_measure,[1,out_measure.shape[0],out_measure.shape[1],1])
out_measure = tf.image.resize_images(out_measure,[img_W,img_H],tf.image.ResizeMethod.BILINEAR) # 将自相关变为与实测自相关相同维度
out_measure = (out_measure - tf.reduce_min(out_measure))/(tf.reduce_max(out_measure) - tf.reduce_min(out_measure))
out_measure = tf.reshape(out_measure,[img_W,img_H])

loss = tf.losses.mean_squared_error(out_measure, raw_measure)

prediction_error = []

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    with tf.variable_scope('train_step'):
        train_op = optimizer.minimize(loss)
    
init_op = (tf.local_variables_initializer(),tf.global_variables_initializer())
saver = tf.train.Saver()
    
with tf.Session() as sess:
    # 神经网络训练准备工作。变量初始化，线程启动
    sess.run(init_op)
#    saver.restore(sess,model_save_path)
    for i in range(test_num):
#        img_ac = plt.imread(ac_save_path  + str(i+11003) + '.png')
        img_ac = plt.imread(ac_save_path)
        img_label = plt.imread(label_path)
        img_ac = np.reshape(img_ac,[1,img_W,img_H,1])
        saver.restore(sess,model_save_path)
#        sess.run(init_op)
        for step in range(200):                
            # new_rand = np.random.uniform(0, 1.0/30.0, size=(1,img_W,img_H,1))      
                    
            if step % 10 == 0:
                # lossval = sess.run(E, feed_dict = {rand: new_rand, x:img_ac, keep_prob:1.0, is_training:True}) 
                lossval,y_pred = sess.run([loss,out], feed_dict = {x:img_ac, keep_prob:1.0, is_training:True})
                
                error = np.sum((img_label/np.max(img_label) - y_pred)**2)/(128*128)
                pred_error = error
                print('[step:%d   measure loss: %f   prediction_loss%f]'%(step,lossval,pred_error))
                
                prediction_error.extend([pred_error])
                
                # new_rand = np.random.uniform(0, 0.0/30.0, size=(1,img_W,img_H,1))
                # image_out = sess.run(out, feed_dict = {rand: new_rand, x:img_ac, keep_prob:1.0, is_training:True}).reshape(img_W,img_H)
                image_out = sess.run(out_pad, feed_dict = {x:img_ac, keep_prob:1.0, is_training:True})
                image_out = 255*(image_out - np.min(image_out))/(np.max(image_out) - np.min(image_out))
                image_out = Image.fromarray(image_out.astype('uint8')).convert('L')  
                image_out = image_out.crop((64,64,192,192))
                image_out.save(result_save_path+'4360_%d.bmp'%(step))
                
            if step % 10 == 0:
                # new_rand = np.random.uniform(0, 0.0/30.0, size=(1,img_W,img_H,1))
                # Ac_compute = sess.run(out_measure, feed_dict = {rand: new_rand, x:img_ac, keep_prob:1.0, is_training:True})
                # Ac_measure = sess.run(raw_measure, feed_dict = {rand: new_rand, x:img_ac, keep_prob:1.0, is_training:True})
                Ac_compute = sess.run(out_measure, feed_dict = {x:img_ac, keep_prob:1.0, is_training:True})
                Ac_measure = sess.run(raw_measure, feed_dict = {x:img_ac, keep_prob:1.0, is_training:True})
                
                auto_out = 255*(Ac_compute - np.min(Ac_compute))/(np.max(Ac_compute) - np.min(Ac_compute))
                auto_out = Image.fromarray(auto_out.astype('uint8')).convert('L') 
                auto_out.save(auto_result_save_path+'4360_%d.png'%(step))
                 
                plt.subplot(141)
                plt.imshow(Ac_measure)             
                plt.subplot(142)
                plt.imshow(Ac_compute) 
                plt.subplot(143)
                plt.imshow(image_out) 
                plt.subplot(144)
                plt.imshow(img_label) 
                plt.title('step:%d'%(step))
                plt.show()
                
                
            # sess.run(train_op, feed_dict = {rand: new_rand, x:img_ac, keep_prob:1.0, is_training:True})
            sess.run(train_op, feed_dict = {x:img_ac, keep_prob:1.0, is_training:True})
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        




