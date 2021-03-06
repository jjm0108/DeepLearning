import numpy as np

# 필터 크기, 스트라이드, 패딩을 고려해 입력 데이터를 2차원 배열로 전개하는 함수
# why? 필터가 적용될 각 영역을 하나의 줄(행)로 전개해 이후에 진행될 합성곱 계산이 편리하도록 입력 데이터를 미리 준비하기 위함
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # N,C에는 패딩을 붙이지 않기 때문에 (0,0)을 해줌
    # H, W에는 패딩을 붙이기 때문에 (pad, pad)를 해줌
    # tuple의 첫번째는 앞단에 붙이는 padding 갯수를 의미하고, 두번째는 뒷단에 붙이는 padding 갯수를 의미함
    # constant는 설정한 pad에서 값이 증가하거나 하지 않고 constant하게 붙여진다는 것을 의미
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

x1 = np.random.rand(1,3,7,7) # (데이터수, 채널 수, 높이, 너비)
col1 = im2col(x1, 5, 5, stride=1, pad=0)

# im2col을 적용한 입력 데이터의 크기 = (필터 적용범위의 총 개수, 필터 하나가 적용되는 범위(필터 하나의 크기))
# 필터 적용범위를 가로행렬로 나타냈을 때 9줄이 나옴
# (높이(7)-필터넓이(5))/스트라이드(1)+1 = 3
# 가로 적용범위가 3이므로 가로세로의 적용범위는 제곱한 9가 됨

# 필터를 세로행렬로 나타냈을 때는 채널수(3) x 높이(5) x 너비(5) = 75줄이 나옴
# 필터의 채널 수는 명시되어있지 않지만, 입력 데이터의 채널 수와 동일해야하기 때문에 3이라 인식

print(col1.shape) #(9, 75)

# 데이터가 1개였던 이전 예제와 달리 데이터가 10개이므로 입력 데이터의 필터 적용범위는 90줄의 가로행렬로 변환됨
x2 = np.random.rand(10,3,7,7) # 데이터 10개
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape) #(90, 75)

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.col = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        # output data의 H,W를 계산
        out_h = int(1+(H + 2* self.pad - FH) / self.stride)
        out_w = int(1+(W + 2* self.pad - FW) / self.stride)

        # 데이터를 전개
        self.col = im2col(x, FH, FW, self.stride, self.pad) # 입력 데이터를 가로 행렬로 전개
        col_W = self.W.reshape(FN, -1).T # 필터를 세로 행렬로 전개 

        # 합성곱 계산
        out = np.dot(self.col, col_W) + self.b # 합성곱 연산, 편향 더함

        # 2차원 행렬로 계산된 결과를 다시 4차원으로 바꿈
        # transpose는 다차원 배열의 축 순서를 바꿔주는 함수
        # (0,3,1,2) 인덱스를 지정하여 축의 순서를 (N,H,W,C)에서 (N,C,H,W)로 변환
        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        # forward 반대로 shape 잡아줌, C가 맨뒤로 오게 transpose해준 다음, FN이 마지막으로 오게 reshape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        # Affine 계층과 동일
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        # 다시 FN이 먼저 나오게 transpose해준 후, reshape
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        # col형태를 다시 image로 돌려놓기
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1+(H - self.pool_h) / self.stride)
        out_w = int(1+(W - self.pool_w) / self.stride)

        # 입력 데이터 전개
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # 각 풀링될 적용범위 행마다 각 열에 풀링될 구성요소들이 배치되어야 함
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # 최대 풀링 (적용 범위 행 중에서 최대값을 갖는 구성요소를 뽑음)
        out = np.max(col, axis=1)

        # 2차원 행렬이 아닌 입력 데이터 차원에 맞게 변형
        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

        return out

class SimpleConvNet:
    def __init__(self, input_dim=(1,28,28), conv_param = {'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}, hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['filter_pad']
        filter_stride = conv_param['filter_stride']

        input_size = input_dim[1]
        # 합성곱계층 계산 결과 크기
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1

        # 풀링계층 결과 크기 (이해가 안되는데)
        pool_output_size = int(filter_num * (conv_output_size/2)) * (conv_output_size/2))

        # 가중치 매개변수 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size) # 필터의 C는 입력데이터의 C와 동일해야하므로 input_dim[0]으로 설정
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size,hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # CNN을 구성하는 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    # forward propagation (predicting)
    def predict(self, x):
        for layer in self.last_layers.values():
            x = layer.forward(x)
        return x

    # forward propagation (label and prediction result comparing)
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y,t)

    def gradient(self, x, t):
        # forward propagation
        self.loss(x,t)

        # backward propagation
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout=layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads
        






