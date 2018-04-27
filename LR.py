# -*- coding: utf-8 -*-
"""
First Attempt Of Machine Learning
Digital recognition
@author: Yezhipeng
"""

from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils

def data_load():
    (X_train, y_train), (X_test, y_test) = mnist.load_data(".mnist.pkl.gz")
    #将数据垂构为二维矩阵，每一行表示一张图中的所有象素，行数为样本数，参考函数reshape
    #为了之后的计算，需要将数据声明为float，参考astype('float32')
    X_train = np.reshape(X_train, [60000, 784])
    X_test = np.reshape(X_test, [10000, 784])
    
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    #将象素数据归一化至范围[0,1]
    X_train = X_train / 255 #灰度图最大值为255
    X_test = X_test / 255
    
    #将y_train,y_test转化为one-hot型数据矩阵，参考函数np_utils.to_categorical
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    
    #计算class数量，每幅图的象素数量
    num_classes = y_train.shape[1]
    num_pixels = X_train.shape[1]
    
    return X_train, y_train, X_test, y_test, num_classes, num_pixels

def sigmoid(z): #请实现sigmoid函数  
    output = 1 / (1 + np.exp(-z))
    return output

def cost(y_train, yhat, num_classes): #计算costfunction，参考函数np.log,sum
    J = -(1/60000) * np.sum([np.dot(y_train[:, 1], np.log(yhat[:, 1])) + np.dot((1 - y_train[:, 1]), np.log(1 - yhat[:, 1])) for i in range(num_classes)])
    return J

def compute(X_train, y_train, w, b, alpha, iteration_num):
    JList=[] #定义一个list,用以记录每一步的cost,检查是否在变小
    for i in range(iteration_num):
        yhat = sigmoid(np.dot(X_train, w) + b) #计算yhat
        J = cost(y_train, yhat, num_classes) #计算cost
        dw = (1/60000) * np.dot(X_train.T, yhat - y_train) #计算dw,参考函数np.dot，注意按样本数进行平均
        db = (1/60000) * (yhat - y_train) #计算db，注意按样本数进行平均
        w = w - alpha * dw #用dw去迭代w,别忘了迭代长度alpha
        b = b - alpha * db #同上，更新b
        #每50次输出一次cost和accurancy，检查梯度下降是否正确
        if (i%50 == 0):
            print ("After " + str(i) + " iterations,the cost is:" + str(J))
            JList.append(J)
            print ("accurancy of train set is:" + str(sum(np.argmax(yhat, axis = 1) == np.argmax(y_train, axis = 1)) / X_train.shape[0]))
        else:
            pass
    return w,b,JList

np.random.seed(1) #定义随机数，保证给w和b进行初始化时，大家的结果是相同的，以使得程序可以对比运行结果，大家不用管它
X_train, y_train, X_test, y_test, num_classes, num_pixels = data_load()
w = np.random.rand(784, 10) - 0.5 #应用np.random.rand初始经w，考虑如何让均值为0
b = np.random.rand(60000, 10) - 0.5 #同上，初始经b，均值为0
alpha = 0.03
iteration_num = 1000
[w, b, JList] = compute(X_train, y_train, w, b, alpha, iteration_num) #应用你完成的compute函数来计算上述参数吧
#用测试集试试，看看新的数据，预测效果怎么样
yhat = sigmoid(np.dot(X_test, w) + b)
print("accurancy of test set is:" + str(sum(np.argmax(yhat, axis = 1) == np.argmax(y_train, axis = 1))/60000))