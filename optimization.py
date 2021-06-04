import numpy as np

# Stochastic Gradient Descent (SGD, 확률적 경사하강법)
class SGD:
    def __init__(self,lr =0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# Momentum(모멘텀) : 물리 법칙에서 영감을 받은 optimizing 기법
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        # v는 물체의 속도로 초기화 때는 아무 값도 담지 않고(None), 대신 처음 update()가 호출될 때 params와 같은 구조의 데이터를 딕셔너리 변수로 저장함
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            # momentum * v (velocity) -> 물리에서의 지면 마찰이나 공기 저항처럼 물체가 아무런 힘을 받지 않을 때 서서히 하강시키는 역할을 함
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] +=self.v[key]

# 전체 매개변수의 학습률 값을 일괄적으로 낮추는 것이 아닌, 각각의 매개변수에 적응적으로 학습률을 조정하면서 학습을 진행
class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        delta = 1e-7

        if self.h is None:
            self.h = {}
        
        for key, val in params.items():
            self.h[key] = np.zeros_like(val)

        for key in params.keys():
            # h는 이전의 기울기 값을 제곱하여 계속 더해줌, 그 이후 매개변수를 갱신할 때 1/\sqrt h를 곱해 학습률을 조정함
            self.h[key] += grads[key] * grads[key]
            # delta는 self.h[key]에 0이 들어있을 때, 0으로 나누는 사태를 방지하기 위한 아주 작은 값
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key])+ delta) 

    