import numpy as np

class ConvolutionLayer:
  # A Convolution layer
    def __init__(self, num_filters, inputs_channel, width, height, stride, padding, learning_rate, name):
        """
        num_filters:     卷积核个数
        inputs_channel:  通道个数
        width：          卷积核宽
        height：         卷积核高
        stride：         卷积核步长
        padding：        输入填充宽度
        learning_rate:  学习率
        name:           卷积层名字
        """
        self.num_filters = num_filters
        self.channel = inputs_channel
        self.width = width
        self.height = height
        self.stride = stride
        self.padding = padding
        self.name = name
        self.lr = learning_rate

        # 所有卷积参数构成一个3维矩阵， (num_filters, channel, width, height)
        # 参数随机初始化，除length*width减小方差
        self.weights = np.zeros((self.num_filters, self.channel, self.width, self.height))
        self.bias = np.zeros((self.num_filters,1))
        for i in range(self.num_filters):
            self.weights[i,:,:,:] = np.random.normal(loc=0, scale=np.sqrt(1./(self.channel*self.width*self.height)), size=(self.channel, self.width, self.height))


    def zero_padding(self, inputs, padding_size):
        w, h = inputs.shape[0], inputs.shape[1]
        new_w = 2 * padding_size + w
        new_h = 2 * padding_size + h
        out = np.zeros((new_w, new_h))
        out[padding_size:w+padding_size, padding_size:h+padding_size] = inputs
        return out

    def forward(self, inputs):
        # input size: (C, W, H)
        # output size: (F ,WW, HH)
        C = inputs.shape[0]
        W = inputs.shape[1]+2*self.padding
        H = inputs.shape[2]+2*self.padding
        self.inputs = np.zeros((C, W, H))
        for c in range(inputs.shape[0]):
            self.inputs[c,:,:] = self.zero_padding(inputs[c,:,:], self.padding)
        WW = (W - self.width)//self.stride + 1
        HH = (H - self.height)//self.stride + 1
        feature_maps = np.zeros((self.num_filters, WW, HH))
        for f in range(self.num_filters):
            for w in range(WW):
                for h in range(HH):
                    feature_maps[f,w,h]=np.sum(self.inputs[:,w:w+self.width,h:h+self.height]*self.weights[f,:,:,:])+self.bias[f]

        return feature_maps

    def backward(self, dy):

        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        F, W, H = dy.shape
        for f in range(F):
            for w in range(W):
                for h in range(H):
                    dw[f,:,:,:]+=dy[f,w,h]*self.inputs[:,w:w+self.width,h:h+self.height]
                    dx[:,w:w+self.width,h:h+self.height]+=dy[f,w,h]*self.weights[f,:,:,:]

        for f in range(F):
            db[f] = np.sum(dy[f, :, :])

        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        return dx

    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}

    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias
