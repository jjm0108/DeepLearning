import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 학습을 위한 장치 얻기
# 가능한 경우 GPU와 같은 하드웨어 가속기에서 모델을 학습
# torch.cuda 를 사용할 수 있는지 확인하고 그렇지 않으면 CPU를 계속 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# 클래스 정의하기
# 신경망 모델을 nn.Module 의 하위클래스로 정의하고, __init__ 에서 신경망 계층들을 초기화 
# nn.Module 을 상속받은 모든 클래스는 forward 메소드에 입력 데이터에 대한 연산들을 구현