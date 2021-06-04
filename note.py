from collections import OrderedDict
import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1/(1+np.exp(-x))
        return self.out

    def backward(self,dout):
        dx = dout * self.out * (1-self.out)
        return dx

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.x = None

    def forward(self, x):
        self.x = x
        W, b = self.params
        out = np.matmul(x,W) + b
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.matmul(dout, W.T)
        dw = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = self.softmax(x)
        self.t = t
        loss = self.cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout):
        batch_size = self.y.shape[0]
        dx = (self.y-self.t) / batch_size
        return dx

    def softmax(self, x):
        x = x - np.max(x) # 오버플로 방지
        return np.exp(x) / np.sum(np.exp(x))

    def cross_entropy_error(self, y, t):
        delta = 1e-7
        batch_size = y.shape[0]
        return -np.sum(t*np.log(y+delta)) / batch_size

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = np.random.randn(I,H)
        self.params['b1'] = np.zeros(H)
        self.params['W2'] = np.random.randn(H,O)
        self.params['b2'] = np.zeros(O)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W1'], self.params['b1'])
        self.lastlayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = self.lastlayer.forward(y,t)
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        batch_size = x.shape[0]
        accuracy = np.sum(y==t) / float(batch_size)
        return accuracy

    def gradient(self, x, t):
        dout = self.loss(x,t)
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads


from minst import load_mnist
x_train, t_train, x_test, t_test = load_mnist(normalize=True, one_hot_label=True)









