import numpy as np
class MaxPoolingLayer:
    # A Max Pooling layer .
    def __init__(self, width, height, stride, name):
        self.width = width
        self.height = height
        self.stride = stride
        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        C, W, H = inputs.shape
        new_width = (W - self.width) // self.stride + 1
        new_height = (H - self.height) // self.stride + 1
        out = np.zeros((C, new_width, new_height))
        for c in range(C):
            for w in range(new_width):
                for h in range(new_height):
                    out[c, w, h] = np.max(
                        self.inputs[c, w * self.stride:w * self.stride + self.width, h * self.stride:h * self.stride + self.height])
        return out

    def backward(self, dy):
        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)

        for c in range(C):
            for w in range(0, W, self.width):
                for h in range(0, H, self.height):
                    st = np.argmax(self.inputs[c, w:w + self.width, h:h + self.height])
                    (idx, idy) = np.unravel_index(st, (self.width, self.height))
                    dx[c, w + idx, h + idy] = dy[c, w // self.width, h // self.height]
        return dx

    def extract(self):
        return