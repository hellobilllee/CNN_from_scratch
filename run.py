from tensorflow.examples.tutorials.mnist import input_data
from network import Net
import matplotlib.pyplot as plt
import numpy as np

print('Loadind data......')
# MNIST_data指的是存放数据的文件夹路径，one_hot=True 为采用one_hot的编码方式编码标签
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
num_classes = 10
#load data
train_images = mnist.train.images                #训练集样本
validation_images = mnist.validation.images      #验证集样本
test_images = mnist.test.images                  #测试集样本
#labels
train_labels = mnist.train.labels                #训练集标签
validation_labels = mnist.validation.labels      #验证集标签
test_labels = mnist.test.labels                  #测试集标签

# demo使用一千张图片做训练与测试， 可以使用全部的训练和测试图片，训练时间会比较久一些
print('Preparing data......')
#train_X = (train_images - np.mean(train_images))/np.std(train_images)
#test_X = (test_images - np.mean(test_images))/ np.std(test_images)
train_X = train_images
test_X = test_images
training_data = train_X.reshape(55000, 1, 28, 28)
training_labels = train_labels
testing_data = test_X.reshape(10000, 1, 28, 28)
testing_labels = test_labels


print(training_data.shape, training_labels.shape)
print(testing_data.shape, test_labels.shape)
LeNet = Net()

print('Training Lenet......')
LeNet.train(training_data=training_data,training_label=training_labels,batch_size=32,epoch=3,weights_file="pretrained_weights.pkl")

print('Testing Lenet......')
LeNet.test(data=testing_data,label=testing_labels,test_size=1000)

print('Testing with pretrained weights......')
LeNet.test_with_pretrained_weights(testing_data, testing_labels, 1000, 'pretrained_weights.pkl')