import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.
from numpy import array
import torch
import gc
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import os,sys
sys.path.insert(0, os.getcwd()) 
from data_analysis_module import *

def data_analysis(df):
    print('*'*50)
    print('data : \n{}'.format(df)) 
    print('*'*50)
    # print('\ndata type : \n{}'.format(df.dtypes))
    # print('*'*50)
    print(f'data length: {len(df)}') 
    print('*'*50)

def train_valid_test_split(train_set_split_ratio, valid_set_split_ratio):
    # split dataset to train/valid
    test_set_split_ratio = 1 - train_set_split_ratio - valid_set_split_ratio
    train_set_split = int(len(df)*train_set_split_ratio) if int(len(df)*train_set_split_ratio)%2!=0 else int(len(df)*train_set_split_ratio)+1
    valid_set_split = int(len(df)*(train_set_split_ratio + valid_set_split_ratio))
    test_set_split = int(len(df)*test_set_split_ratio)

    train_set = df[:train_set_split]
    valid_set = df[train_set_split:valid_set_split]
    test_set = df[-test_set_split:]

    print(f'length of train_set : {len(train_set)}')
    print(f'length of valid_set : {len(valid_set)}')

    # show the proportion of each dataset
    print('*'*50)
    print('Proportion of train_set : {:.2f}%'.format(len(train_set)/len(df)))
    print('Proportion of valid_set : {:.2f}%'.format(len(valid_set)/len(df)))
    print('*'*50)

    
    return train_set, valid_set, test_set

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
        self.fc1 = nn.Linear(64,50)
        self.fc2 = nn.Linear(50,1)
        
    def forward(self,x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

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


# data importing
df = ImportFDCData("1")[:50000]
data_analysis(df)

pred_param = 'Ch 1 Onboard Cryo Temp 1st Stage'

train_set, valid_set, test_set = train_valid_test_split(train_set_split_ratio = 0.4, valid_set_split_ratio =0.2)

plt.figure(figsize=(16,8), dpi=100)
plt.title("data split")
plt.plot(train_set['time'], train_set[pred_param], label="train_set")
plt.plot(valid_set['time'], valid_set[pred_param], label="valid_set")
plt.plot(test_set['time'], test_set[pred_param], label="test_set")
plt.legend(loc="upper right")
plt.show()
plt.pause(1)
plt.close()

n_steps = 3
train_x,train_y = split_sequence(train_set[pred_param].values,n_steps)
valid_x,valid_y = split_sequence(valid_set[pred_param].values,n_steps)

# summarize the data
print("train_x, train_y splitted  : ")
for i in range(10):
	print(train_x[i], train_y[i])
print('*'*50)
print(f'length of train_x, train_y : {len(train_x)} , {len(train_y)}')
print(f'length of valid_x, valid_y : {len(valid_x)} , {len(valid_y)}')
print('*'*50)


# ????????? ?????? ?????? ??????
# ????????? ?????? GPU??? ?????? ???????????? ??????????????? ????????? ??????
# torch.cuda ??? ????????? ??? ????????? ???????????? ????????? ????????? CPU??? ?????? ??????
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model, optimizer, criterion ??????
model = CNN_ForecastNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # Adam
criterion = nn.MSELoss() # Mean Squared Error 

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1 # for univariate series, the number of features is one, for one variable.
train = MyDataset(train_x.reshape(train_x.shape[0],train_x.shape[1],n_features),train_y)
valid = MyDataset(valid_x.reshape(valid_x.shape[0],valid_x.shape[1],n_features),valid_y)

train_loader = torch.utils.data.DataLoader(train,batch_size=1,shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid,batch_size=1,shuffle=False)

torch.set_default_tensor_type(torch.DoubleTensor)

train_losses = []
valid_losses = []

epochs = 100
for epoch in range(epochs):
    print('epochs {}/{}'.format(epoch+1,epochs))
    Train()
    Valid()
    gc.collect()

    if epoch%10==0:
        plt.plot(train_losses,label='train_loss')
        plt.plot(valid_losses,label='valid_loss')
        plt.title(f'Epoch {epoch}\'s MSE Loss')
        plt.ylim(0, 1)
        plt.legend(loc='upper left')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

path = './model/'
os.makedirs(path, exist_ok=True)
filename = f'FDC data {pred_param} prediction model'
filename = os.path.join(path, filename)
torch.save(model, filename+".pt")


plt.plot(train_losses,label='train_loss')
plt.plot(valid_losses,label='valid_loss')
plt.title(f'Final MSE Loss (epoch {epochs})')
plt.xlim(0, epochs)
plt.ylim(0, 1)
plt.legend(loc='upper left')

test_x , test_y = split_sequence(test_set[pred_param].values,n_steps)
inputs = test_x.reshape(test_x.shape[0],test_x.shape[1],n_features)
print('*'*50)
print(f'length of test_x, test_y : {len(test_x)} , {len(test_y)}')
print('*'*50)

model.eval()
prediction = []
batch_size = 2
iterations =  int(inputs.shape[0])

for i in range(iterations):
    preds = model(torch.tensor([inputs[i]]).float()).detach().item()
    prediction.append(preds)

plt.figure(figsize=(16,8), dpi=100)
plt.title("result")
plt.plot(prediction, label="predicted one")
plt.plot(test_y, label="real one")
plt.legend(loc="upper right")
plt.show()

# anomaly_score = np.array(prediction)-np.array(test_y)

# fig, ax = plt.subplots(3, 1,figsize=(11,6))
# ax[0].set_title("result")
# ax[0].plot(test_y, label="real one")
# ax[1].plot(prediction, label="predicted one")
# ax[2].plot(anomaly_score, label="target-prediction")

# plt.show()

