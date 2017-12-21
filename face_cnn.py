import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image


#编写cnn网络
class NetWork(object):
    def __init__(self):
        self.x=tf.placeholder(dtype=tf.float32,shape=(None,250,250,1),name='x-input')
        self.y=tf.placeholder(dtype=tf.float32,shape=(None,1,2),name='y-input')
        self.keep_prob=tf.placeholder(tf.float32)
        self.sess=tf.Session()

    def getImgData(self,img):
        img=self.resizeImg(img)
        imgData=np.asarray(img)
        imgData=imgData.mean(imgData,-1)
        imgData=imgData/255
        return imgData

    def resizeImg(self,img):
        img.resize((250,250))
        return img

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(
            x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'
        )
    def cnn(self,x):
        W_conv1 = self.weight_variable([5,5,1,32])
        b_conv1 = self.bias_variable([32])
        x_image = tf.reshape(x, [-1,250,250,1])
        h_conv1=tf.nn.relu(self.conv2d(x_image,W_conv1)+b_conv1)
        h_pool1=self.max_pool_2x2(h_conv1)

        W_conv2=self.weight_variable([5,5,32,64])
        b_conv2=self.bias_variable([64])
        h_conv2=tf.nn.relu(self.conv2d(h_pool1,W_conv2)+b_conv2)
        h_pool2=self.max_pool_2x2(h_conv2)

        #print(h_pool2.shape)(? 63 63 64)

        W_fc1=self.weight_variable([63*63*64,1024])
        b_fc1=self.bias_variable([1024])
        h_pool2_flat=tf.reshape(h_pool2,[-1,63*63*64])
        h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

        h_fc1_drop=tf.nn.dropout(h_fc1,self.keep_prob)

        W_fc2=self.weight_variable([1024,2])
        b_fc2=self.bias_variable([2])

        y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2

        return y_conv

