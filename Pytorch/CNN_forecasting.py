import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from numpy import array
import torch
import gc
import torch.nn as nn
from torch.utils.data import Dataset

# import datset
df = pd.read_csv('./Pytorch/PV_Elec_Gas3.csv').rename(columns={'date':'timestamp'}).set_index('timestamp')
# for param in df.columns:
#     df[param] = df[param].astype('float32').dtypes

print('*'*50)
print('data : \n{}'.format(df)) 
print('*'*50)
print('\ndata type : \n{}'.format(df.dtypes))
print('*'*50)

# split dataset to train/valid
train_set = df[:'31/10/2018']
valid_set = df['1/11/2018':'18/11/2019']

# show the proportion of each dataset
print('Proportion of train_set : {:.2f}%'.format(len(train_set)/len(df)))
print('Proportion of valid_set : {:.2f}%'.format(len(valid_set)/len(df)))

# The split_sequence() function below will split a given univariate sequence into multiple samples 
# where each sample has a specified number of time steps and the output is a single time step.
def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        
        end_ix = i + n_steps
        
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

n_steps = 3
train_x,train_y = split_sequence(train_set['Gas/day'].values,n_steps)
valid_x,valid_y = split_sequence(valid_set['Gas/day'].values,n_steps)

# summarize the data
print('*'*50)
print("train_x, train_y splitted  : ")
for i in range(10):
	print(train_x[i], train_y[i])
print('*'*50)

class MyDataset(Dataset):
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        
        return item,label

# Conv - Relu - FC - FC
class CNN_ForecastNet(nn.Module):
    def __init__(self):
        super(CNN_ForecastNet,self).__init__()
        self.conv1d = nn.Conv1d(n_steps,64,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64*2,50)
        self.fc2 = nn.Linear(50,1)
        
    def forward(self,x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

# 학습을 위한 장치 얻기
# 가능한 경우 GPU와 같은 하드웨어 가속기에서 모델을 학습
# torch.cuda 를 사용할 수 있는지 확인하고 그렇지 않으면 CPU를 계속 사용
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model, optimizer, criterion 설정
model = CNN_ForecastNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # Adam
criterion = nn.MSELoss() # Mean Squared Error 

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1 # for univariate series, the number of features is one, for one variable.
train = MyDataset(train_x.reshape(train_x.shape[0],train_x.shape[1],n_features),train_y)
valid = MyDataset(valid_x.reshape(valid_x.shape[0],valid_x.shape[1],n_features),valid_y)

train_loader = torch.utils.data.DataLoader(train,batch_size=2,shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid,batch_size=2,shuffle=False)

torch.set_default_tensor_type(torch.DoubleTensor)

train_losses = []
valid_losses = []

def Train():
    running_loss = .0
    model.train()
    
    for idx, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device)
    
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(inputs.float())
        loss = criterion(preds,labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss
        
    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss.detach().numpy())
    
    print(f'train_loss {train_loss}')
    
def Valid():
    running_loss = .0
    
    model.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs.float())
            loss = criterion(preds,labels)
            running_loss += loss
            
        valid_loss = running_loss/len(valid_loader)
        valid_losses.append(valid_loss.detach().numpy())
        print(f'valid_loss {valid_loss}')

epochs = 200
for epoch in range(epochs):
    print('epochs {}/{}'.format(epoch+1,epochs))
    Train()
    Valid()
    gc.collect()

import matplotlib.pyplot as plt
plt.plot(train_losses,label='train_loss')
plt.plot(valid_losses,label='valid_loss')
plt.title('MSE Loss')
plt.ylim(0, 100)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

target_x , target_y = split_sequence(train_set.Elec_kW.values,n_steps)
inputs = target_x.reshape(target_x.shape[0],target_x.shape[1],1)

model.eval()
prediction = []
batch_size = 2
iterations =  int(inputs.shape[0]/2)

for i in range(iterations):
    preds = model(torch.tensor(inputs[batch_size*i:batch_size*(i+1)]).float())
    prediction.append(preds.detach().numpy())

fig, ax = plt.subplots(1, 2,figsize=(11,4))
ax[0].set_title('predicted one')
ax[0].plot(prediction)
ax[1].set_title('real one')
ax[1].plot(target_y)
plt.show()
