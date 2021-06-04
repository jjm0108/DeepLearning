import numpy as np
from collections import OrderedDict

# 출력층에 항등 함수를 사용할 때는, 오차제곱합을 손실함수로 쓰면 역전파의 식이 깔끔하게 떨어진다
def sum_squares_error(y,t):
    return 0.5* np.sum((y-t)**2)

class Relu:
    def __init__(self):
        self.mask = None
        self.params = []

    def forward(self, x):
        self.mask = (x<=0) # mask는 Boolean 형식의 numpy 배열
        out = x.copy()
        # relu 함수의 모양을 생각해보자
        # 0 이하의 숫자는 모두 0으로 표현, 그 이상은 그대로
        out[self.mask] = 0
        return out
    
    # forward prop때 입력 값이 0이하면 backprop 때의 값은 0이 되어야 함
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

# Sigmoid 계층 
class Sigmoid:
    def __init__(self):
        # Sigmoid 계층에는 학습하는 매개변수가 따로 없으므로 params, grads는 빈 리스트로 초기화
        self.params, self.grads = [], [] 

        # forwardprop 때는 output을 self.out에 저장
        # backprop을 계산할 때 이 self.out 변수를 사용
        self.out = None

    def forward(self, x):
        out = 1 / (1+np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        # sigmoid 함수의 미분값은 s(x)*(1-s(x)) (손으로 계산해보면 나옴)
        # 따라서 dout 값에 sigmoid 함수의 미분값을 곱한 값이 dx가 됨 (chain rule)
        dx = dout * self.out * (1.0 - self.out)
        return dx

class Affine:
    def __init__(self, W, b):
        self.params = [W,b] # 초기화될 때 가중치와 편향을 입력받아 순전파시 학습에 사용
        # grads : 기울기를 보관하는 리스트
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
    
    def forward(self, x):
        W, b = self.params
        out = np.matmul(x,W) + b
        self.x = x
        return out

    # Affine의 역전파는!! 배열의 모양을 만들기위한 블럭 찾기이다!! (값은 신경쓰지 않음, 역전파를 미분으로 생각하면 안될듯)
    # 행렬 곱연산의 역전파는 단순히 벡터 사이즈만 맞춰주면 된다 
    def backward(self, dout):
        W, b = self.params
        # 안되면 외우자
        dx = np.matmul(dout, W.T) # 벡터 사이즈를 맞춰주기 위해 전치행렬을 곱해주는 것
        dw = np.matmul(self.x.T, dout)

        # 예를 들어 기존의 (10,) 사이즈였던 bias가 계산을 위해 배치 크기 N만큼 늘어나 N x 10 크기로 계산이 되었기 때문에 
        # 역전파 진행 시 강제로 np.sum(dout, axis=0)을 통해서 다시 (10,) 사이즈의 벡터로 바꿔주는 것이다
        db = np.sum(dout, axis=0) # asix=0은 열을 기준으로 합치는 것

        self.grads[0][...] = dw
        self.grads[1][...] = db

        return dx

# softmax 함수의 loss function으로 cross entropy error를 사용하면 역전파가 말끔히 떨어짐
# softmax는 단순히 y-t이다
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmax의 출력
        self.t = None # 정답 레이블 (one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

    def softmax(x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 

        x = x - np.max(x) # 오버플로 대책 (오버플로: 지수함수에서 너무 큰 값을 출력하는 것을 방지하기 위함)
        return np.exp(x) / np.sum(np.exp(x))

    def cross_entropy_error(y,t):
        # 신경망의 출력(y)가 1차원 벡터일 때는 데이터의 shape을 설정해주어야 함
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]

        # np.log에 0을 입력해 마이너스 무한대(-inf)값을 출력하지 않도록 아주 작은 값을 더해줌
        delta = 1e-7 
        return -np.sum(t*np.log(y[np.arrange(batch_size),t]+delta)) / batch_size # cross entropy 손실함수의 평균을 출력

# Stochastic Gradient Descent (SGD, 확률적 경사하강법)
class SGD:
    def __init__(self,lr =0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# Dropout(드롭아웃) : 데이터 훈련 시(training) 은닉층의 뉴런을 무작위로 골라 삭제하면서 학습하는 방법
# 삭제된 뉴런은 신호를 전달하지 못함
# test를 진행할 때는 뉴런을 삭제하지 않음
class Dropout:
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flag=True):
        # training
        if train_flag:
            # np.random.rand(*x.shape)은 x와 크기가 같은(*x.shape) [0,1] 범위에서 균일한 분포를 가지는(rand) 난수 행렬을 생성
            # 생성한 난수 행렬 중 drop_out_raio보다 작은 값만 False이고, 나머지는 True인 행렬
            # 쉽게 말해서, x가 (2,3)행렬일 때, dropout_ratio가 0.1이라면 
            # self.mask = array([[ True,  True,  True],[ True,  True, False]])
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask

        # test    
        else:
            # test때는 각 뉴런의 출력에 훈련 때 삭제 안한 비율을 곱해서 출력
            return x * (1- self.dropout_ratio)

    # 순전파 때 신호를 통과시키는 뉴런은 역전파 때도 신호를 그대로 통과시키고, 순전파 때 통과시키지 않은 뉴런은 역전파때도 신호를 차단
    def backward(self, dout):
        return dout * self.mask
    


# Affine-Sigmoid-Affine
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        I, H, O = input_size, hidden_size, output_size

        # 가중치와 편향 초기화 (Xavier 초깃값(선형 activation function 사용), He 초깃값(Relu사용))
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(I,H)
        self.params['b1'] = np.zeros(H)
        self.params['W2'] = weight_init_std * np.random.randn(H,O)
        self.params['b2'] = np.zeros(O)

        # node_num = 100 # 앞 층의 노드 수
        # Xavier 초깃값 (활성화값을 광범위하게 분포시킬 목적으로 앞 계층의 노드가 n개 일때, 1/sqrt(n) 분포를 사용)
        # w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
        # He 초깃값 (Relu는 음의 영역이 0이기 때문에 더 넓게 분포시키기 위해 2배의 계수 sqrt(2/n)을 사용)
        # w = np.random.randn(node_num, node_num) * np.sqrt(2/node_num)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W1'], self.params['b1'])
        self.lastlayer = SoftmaxWithLoss()

        # 각 계층의 모든 가중치를 params 리스트에 모은다
        self.params = []
        for layer in self.layers.values():
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers.values():
            x= layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.lastlayer.forward(y,t)
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1) # 예측한 클래스가 무엇인지 index를 불러옴
        t = np.argmax(t, axis=1) # 정답(라벨)의 클래스

        if t.ndim != 1 : t = np.argmax(t, axis=1)
        batch_size = x.shape[0]
        accuracy = np.sum(y==t) / float(batch_size) # 예측한 클래스가 정답과 일치하는 확률의 평균
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
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 신경망 네트워크 정의
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
optimizer = SGD()

# 하이퍼파라미터 정의
iters_num = 10000 # 반복 횟수
train_size = x_train.shape[0] # N
batch_size = 100 # 무작위로 뽑을 미니배치 크기
learning_rate = 0.1

# 학습 경과에 따라 loss function이 어떻게 감소하는지 평가하기 위함
train_loss_list = []
# 학습 경과에 따라 accuracy가 어떻게 증가하는지 평가하기 위함
train_acc_list = []
test_acc_list = []

# 1 에폭당 몇 번의 학습을 진행하는지 계산
# 에폭 epoch : 1에폭은 미니배치 학습을 여러번 진행해 학습에서 훈련 데이터를 모두 소진했을 때의 횟수에 해당한다. 
# 즉, 훈련 데이터 1000개를 100개의 미니배치로 학습할 경우, 확률적 경사 하강법을 10회 반복하면 모든 훈련 데이터를 소진한게 되므로, 10회가 1에폭이 된다. 
iter_per_epoch = max (train_size / batch_size , 1 )


for i in range(iters_num):
    # 미니배치 획득 (batch_size 갯수 만큼 train set의 index를 뽑아낸다)
    batch_mask = np.random.choice(train_size, batch_size)

    # 위 코드에서 무작위로 뽑은 index에 맞는 학습 데이터(x)와 라벨(t)를 미니배치 데이터(x_batch, t_batch)로 만듬
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산 (loss 함수 최소값을 위한 방향성)
    grads = network.gradient(x_batch, t_batch)

    # 매개변수 갱신
    optimizer.update(grads)

    # 학습 경과(학습 횟수가 늘어가면서 loss 함수가 어떻게 변화하는지) 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"train accuracy, test accuracy | {train_acc} , {test_acc}")
