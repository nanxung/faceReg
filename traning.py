import tensorflow as tf
import pandas as pd
import numpy as np
from face_cnn import NetWork
import os
from PIL import Image
import random


def getImgData(img):
    img = resizeImg(img)
    imgData = np.asarray(img)
    # print(imgData.shape)
    imgData = np.mean(imgData, -1)
    imgData = imgData / 255

    return imgData


def resizeImg(img):
    img=Image.open(img)
    img=img.resize((250, 250))
    return img

def getDataLs():
    fl=['./img/cb/'+ str(i) for i in os.listdir('./img/cb/')]
    fl.extend(['./img/jzh/'+str(i) for i in os.listdir('./img/jzh/')])
    return fl
def getData(batch_size):
    trainData=[]
    labelData=[]
    fl=getDataLs()
    dls=[random.choice(fl) for i in range(batch_size)]
    for d in dls:
        s=d.split('/')
        if 'cb' == s[2]:
            trainData.append(getImgData(d))
            labelData.append([1,0])
        elif 'jzh' == s[2]:
            trainData.append(getImgData(d))
            labelData.append([0,1])
    return np.asarray(trainData),np.asarray(labelData)


def train():
    net=NetWork()
    y_=net.cnn(net.x)
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_,labels=net.y))
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(net.y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    net.sess.run(tf.global_variables_initializer())
    for i in range(20000):
        trainData,labelData=getData(1)
        if i%100==0 & i!=0:
            print(trainData.shape)
            train_accuracy=net.sess.run(accuracy,feed_dict={
                net.x:np.reshape(trainData,[-1,250,250,1]),
                net.y: np.reshape(labelData,[-1,1,2]),
                net.keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        print(i)
        net.sess.run(train_step,feed_dict={net.x: np.reshape(trainData,[-1,250,250,1]),
                                               net.y: np.reshape(labelData,[-1,1,2]),
                                               net.keep_prob: 0.5})
    print("test accuracy %g"%net.sess.run(train_step,feed_dict={
        net.x: np.reshape(trainData, [-1, 250, 250, 1]),
        net.y: np.reshape(labelData, [-1, 1,2]),
        net.keep_prob: 1.0}))

if __name__=='__main__':
    train()