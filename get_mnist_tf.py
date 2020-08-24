from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# MNIST_data指的是存放数据的文件夹路径，one_hot=True 为采用one_hot的编码方式编码标签
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#load data
train_X = mnist.train.images                #训练集样本
validation_X = mnist.validation.images      #验证集样本
test_X = mnist.test.images                  #测试集样本
#labels
train_Y = mnist.train.labels                #训练集标签
validation_Y = mnist.validation.labels      #验证集标签
test_Y = mnist.test.labels                  #测试集标签

print(train_X.shape,train_Y.shape)          #输出训练集样本和标签的大小

#查看数据，例如训练集中第一个样本的内容和标签
#print(train_X[0])       #是一个包含784个元素且值在[0,1]之间的向量
#print(train_Y[0])

#可视化样本，下面是输出了训练集中前20个样本
figure,axes = plt.subplots(2, 3, figsize=[40,20])
axes = axes.flatten()
for i in range(6):
    img = train_X[i].reshape(28, 28)
    axes[i].imshow(img,cmap='Greys')
axes[0].set_xticks([])
axes[0].set_yticks([])
plt.tight_layout()
plt.show()