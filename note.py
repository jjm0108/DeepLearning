import collections
from CNN import im2col
import numpy as np

class Convolution:
    def __init__(self, W, b, stride, pad):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_W = None

    def forward(self, x):
        N, C, H, W = self.x.shape
        FN, C, FH, FW = self.W.shape

        out_H = (H + 2*self.pad - FH + 1) / self.stride
        out_W = (W + 2*self.pad - FW + 1) / self.stride

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(FN, FH, FW, -1).transpose(0,3,1,2)
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape

        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        db = np.sum(dout, axis=0)
        dW = np.dot(self.col.T, dout)
        dW = dW.transpose(1,0).reshape(FN,C,FH,FW)
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx