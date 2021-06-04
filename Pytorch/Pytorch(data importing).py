# TorchVision 에서 Fashion-MNIST 데이터셋을 불러오는 예제
# Fashion-MNIST는 Zalando의 기사 이미지 데이터셋
# training set : 60,000개
# test set : 10,000개
# 각 예제는 흑백(grayscale)의 28x28 이미지와 10개 분류(class) 중 하나인 정답(label)으로 구성됨

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

# training data importing
training_data = datasets.FashionMNIST(
    root="data", # root : 학습/테스트 데이터가 저장되는 경로
    train=True, # train : 학습용 또는 테스트용 데이터셋 여부를 지정
    download=True, # download=True : root 에 데이터가 없는 경우 인터넷에서 다운로드
    # transform이란?
    # 데이터가 항상 머신러닝 알고리즘 학습에 필요한 최종 처리가 된 형태로 제공되지는 않기 때문에 변형(transform)을 해서 데이터를 조작하고 학습에 적합하게 만든다.
    # 모든 TorchVision 데이터셋들은 변형 로직을 갖는, 호출 가능한 객체(callable)를 받는 매개변수 두개를 가짐
    # transform : 특징(feature)을 변경
    # target_transform : 정답(label)을 변경
    # torchvision.transforms 모듈은 주로 사용하는 몇가지 변형(transform)을 제공함

    # 예를 들어, FashionMNIST 데이터는 PIL Image 형식이며, 정답(label)은 정수(integer)이기 때문에
    # 학습을 하려면 정규화(normalize)된 텐서 형태의 특징(feature)과 원-핫(one-hot)으로 부호화(encode)된 텐서 형태의 정답(label)이 필요하다
    # 이러한 변형(transformation)을 하기 위해 ToTensor 와 Lambda 를 사용한다
    # ToTensor() 는 PIL Image나 NumPy ndarray 를 FloatTensor 로 변환하고, 이미지의 픽셀의 크기(intensity) 값을 [0., 1.] 범위로 비례하여 조정(scale)
    # Lambda 변형은 사용자 정의 람다(lambda) 함수를 적용
    transform=ToTensor(), # transform 과 target_transform : 특징(feature)과 정답(label) 변형(transform)을 지정
    # 정수를 원-핫으로 부호화된 텐서로 바꾸는 함수를 정의
    # 먼저 (데이터셋 정답의 개수인) 크기 10짜리 zero tensor를 만들고, scatter_ 를 호출하여 주어진 정답 y 에 해당하는 인덱스에 value=1 을 할당합니다.
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# test data importing
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# 정답(label)에 대한 map
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
# 학습 데이터 일부를 시각화
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# 파일에서 사용자 정의 데이터셋 만들기
# 사용자 정의 Dataset 클래스는 반드시 3개 함수를 구현해야 한다: __init__, __len__, and __getitem__. 

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        # 이미지와 주석 파일(annotation_file)이 포함된 디렉토리와 두가지 변형(transform)을 초기화
        self.img_labels = pd.read_csv(annotations_file) # 정답은 annotations_file csv 파일에 별도로 저장
        
        self.img_dir = img_dir # 이미지들은 img_dir 디렉토리에 저장됨
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # 데이터셋의 샘플 개수를 반환
        return len(self.img_labels)

    def __getitem__(self, idx):
        # 주어진 인덱스 idx 에 해당하는 샘플을 데이터셋에서 불러오고 반환
        # 인덱스를 기반으로, 디스크에서 이미지의 위치를 식별
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # self.img_labels.iloc[idx, 0]는 이미지 이름
        image = read_image(img_path) # read_image 를 사용하여 이미지를 텐서로 변환
        label = self.img_labels.iloc[idx, 1] # self.img_labels 의 csv 데이터로부터 해당하는 정답(label)을 가져옴

        # (해당하는 경우) 변형(transform) 함수들을 호출한 뒤, 텐서 이미지와 라벨을 Python 사전(dict)형으로 반환
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        sample = {"image": image, "label": label}

        return sample

# DataLoader로 학습용 데이터 준비하기
# Dataset 은 데이터셋의 특징(feature)을 가져오고 하나의 샘플에 정답(label)을 지정하는 일을 한 번에 한다
# 모델을 학습할 때, 일반적으로 샘플들을 “미니배치(minibatch)”로 전달하고, 
# 매 에폭(epoch)마다 데이터를 다시 섞어서 과적합(overfit)을 막고, 
# Python의 multiprocessing 을 사용하여 데이터 검색 속도를 높이려고 한다
# DataLoader 는 간단한 API로 이러한 복잡한 과정들을 추상화한 반복 가능한 객체(iteratable)이다.

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True) # shuffle=True 로 지정했으므로, 모든 배치를 반복한 뒤 데이터가 섞임
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# DataLoader를 통해 반복하기(iterate)
# DataLoader 에 데이터셋을 불러온 뒤에는 필요에 따라 데이터셋을 반복(iterate)할 수 있다. 
# 아래의 각 반복(iteration)은 (각각 batch_size=64 의 특징(feature)과 정답(label)을 포함하는) train_features 와 train_labels 의 묶음(batch)을 반환합니다. 
train_features, train_labels = next(iter(train_dataloader))

# 이미지와 정답(label)을 표시합니다.
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
