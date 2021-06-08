#-*- coding:utf-8 -*-

# 실행방법 : python or python3 iris_mlp.py(같은 경로에 iris.csv 있어야함)

# environment
# pandas 1.2.3, numpy 1.19.2, scikit-learn 0.24.1, pytorch 1.7.1, torchsummary 1.5.1
# matplotlib 3.3.4, seaborn 0.11.1, shutil

import os
import shutil
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchsummary import summary

#CUDA : NVIDIA가 GPU를 이용한 프로그래밍이 가능하도록 만든 것
#GPU를 활용하면 많은 양의 연산을 동시에 처리할 수 있기 때문에 torch.device를 통해 gpu의 사용이 가능한지 여부를 확인하는 것
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SetNetworkConfig():
    '''
    Definition of network's hyper parameter for model

    This class sets model_info structure
    user_defined_name: Dataset or project name
    batch_size: Number of batch data 
    initial_lr: Initial learning rate 
    var_decay: Learning rate decay
    epochs: Number of iteration for training total train data
    classes: Number of class about dataset
	'''

    def __init__(
        self,
        user_defined_name='',
        batch_size = 10,
        initial_lr = 0.01,
        var_decay = 0.5,
        epochs = 50,
        classes = 3): 
        
        self.user_defined_name = user_defined_name
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.var_decay = var_decay
        self.epochs = epochs
        self.classes = classes

    def __call__(self):

        cfg = dict()
        cfg['user_defined_name'] = self.user_defined_name
        cfg['batch_size'] = self.batch_size
        cfg['initial_lr'] = self.initial_lr
        cfg['var_decay'] = self.var_decay
        cfg['epochs'] = self.epochs
        cfg['classes'] = self.classes
        
        return cfg


def data_analysis(df, save_folder):
    
    print('data : \n{}'.format(df))
    print('*'*50)
    print('data shape : {}'.format(df.shape))
    print('\ndata columns : \n{}'.format(df.columns))
    print('\nspecies count : \n{}'.format(df['species'].value_counts()))
    print('\ndata type : {}'.format(type(df)))

    print('*'*50)
    print('data null check : \n{}'.format(df.isnull().sum()))
    
    print('*'*50)
    species_mean = df.groupby(df['species']).mean()
    print('data mean : \n{}'.format(species_mean))

    species_mean.T.plot.bar(rot=0)
    plt.title('species means of features')
    plt.xlabel('features')
    plt.ylabel('mean')
    plt.savefig(os.path.join(save_folder, 'problem5_species_mean.png'), bbox_inches='tight', pad_inches=0.0)
    #plt.show()


    sns.pairplot(df, hue='species', size=3)
    plt.savefig(os.path.join(save_folder, 'problem5_data_scatter.png'), bbox_inches='tight', pad_inches=0.0)
    #plt.show()


def data_preprocessing():

    data = pd.read_csv("iris.csv").rename(columns={'variety':'species'})    
    
    ## expected purpose 
    supervised_df = data['species']

    labels = data['species'].unique().tolist()
    print('labels : ', labels)

    index_to_label_dict = {i:label for i, label in enumerate(labels)}
    print(index_to_label_dict)

    data_x = data.drop(['species'], axis=1).values
    data_y = supervised_df.map(lambda x: labels.index(x)).values

    ## split train and valid dataset(based on y_data)
    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, train_size=0.8, test_size=0.2)

    # print(train_x)
    # print(valid_x)
    # print(train_y)
    # print(valid_y)
    
    return (train_x, train_y), (valid_x, valid_y), index_to_label_dict


def accuracy(output, target):

    batch_size = target.size(0)    

    _, pred = output.float().topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = correct[0].view(-1).float().sum(0) / batch_size

    return acc


def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth', save_all=False):
    filename = os.path.join(path, filename)
    torch.save(state, filename)

    print('is_best : ', is_best)
    if is_best:
        if save_all:
            shutil.copyfile(filename, os.path.join(
                path, 'checkpoint_epoch_%s.pth' % state['epoch']))
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth'))


class Problem4Dataset(Dataset):

    def __init__(self, dataset):
        
        self.dataset_x = dataset[0]
        self.dataset_y = dataset[1]


    def __len__(self):
        return len(self.dataset_x)


    def __getitem__(self, idx):
        
        data_x = self.dataset_x[idx]
        data_y = np.array(self.dataset_y[idx])
        
        x2tensor = torch.from_numpy(data_x).float()
        y2tensor = torch.from_numpy(data_y).long()

        return (x2tensor, y2tensor)


class NetworkBase(nn.Module):
    def __init__(self,
                 network_config, 
                 input_shape, 
                 *args, **kargs):
        self.network_optimizer = None
        self.network_loss = None
        self.network_config = network_config
        self.input_shape = input_shape
        self.log_save_path = None
        self.cp_save_path = None
        self.result_log = None

        super().__init__(*args, **kargs)

    def fit(self, train_loader, valid_loader=None, verbose=1, lr_scheduler=None, save_path=None):

        self.train()

        if not save_path is None:

            os.makedirs(save_path, exist_ok=True)

            self.log_save_path = os.path.join(save_path, 'results.csv')
            self.result_log = ResultsLog(path=self.log_save_path)

            self.cp_save_path = os.path.join(save_path, 'model')

            best_tr_ps1 = 0            
            best_va_ps1 = 0

        for epoch in range(self.network_config.epochs):

            losses = AverageMeter()
            top1 = AverageMeter()

            if self.network_optimizer is not None:
                for param_group in self.network_optimizer.param_groups:
                    lr = param_group['lr']

                if verbose == 1:
                    print('\nEpoch: %d Learning Rate: %f' % (epoch, lr))


            for i, (inputs, target) in enumerate(train_loader):


                if torch.cuda.is_available():

                    inputs = inputs.cuda()
                    target = target.cuda()

                outputs = self.forward(inputs)

                epoch_loss = self.network_loss(outputs, target)

                self.network_optimizer.zero_grad()
                epoch_loss.backward()


                self.network_optimizer.step()

                pred = accuracy(outputs.data, target)
                losses.update(epoch_loss.item(), inputs.size(0))
                top1.update(pred.item(), inputs.size(0))

            train_loss = losses.avg
            train_ps1 = top1.avg
            tr_is_best = (train_ps1 >= best_tr_ps1)
            best_tr_ps1 = max(train_ps1, best_tr_ps1)            

            print('train_loss : ', train_loss)
            print('train_ps1 : ', train_ps1)

            if not valid_dataset is None:
                valid_loss, valid_ps1 = self.evaluate(valid_loader)

            if not save_path is None:
                # remember best validation predict1 and checkpoint
                if best_va_ps1 == 1.0:
                    va_is_best = (valid_ps1 >= best_va_ps1)
                    best_va_ps1 = max(valid_ps1, best_va_ps1)

                    if tr_is_best == True and va_is_best == True:
                        is_best = True
                    else:
                        is_best = False

                else:
                    is_best = (valid_ps1 >= best_va_ps1)
                    best_va_ps1 = max(valid_ps1, best_va_ps1)

                save_checkpoint({
                    'epoch' : epoch,
                    'model' : self,
                    'state_dict' : self.state_dict(),
                    'optimizer_state_dict()' : self.network_optimizer.state_dict(),
                    'loss_func' : self.network_loss,
                    'tr_loss' : train_loss,
                    'va_loss' : valid_loss,
                    'tr_pred1' : train_ps1,
                    'va_pred1' : valid_ps1,
                    'best_tr_pred1': best_tr_ps1,
                    'best_va_pred1': best_va_ps1
                    }, is_best, path=save_path, save_all=False)
                
                self.result_log.add(epoch=epoch, best_tr_pred1=best_tr_ps1, best_va_pred1=best_va_ps1,
                               train_loss=train_loss, train_pred1=train_ps1,
                               valid_loss=valid_loss, valid_pred1=valid_ps1)
                self.result_log.save()


            lr_scheduler.step()

    
    def evaluate(self, valid_loader):

        self.eval()

        losses = AverageMeter()
        top1 = AverageMeter()

        for i, (inputs, target) in enumerate(valid_loader):

            with torch.no_grad():

                if torch.cuda.is_available():

                    inputs = inputs.cuda()
                    target = target.cuda()

            outputs = self.forward(inputs)
            
            epoch_loss = self.network_loss(outputs, target)

            pred = accuracy(outputs.data, target)
                
            losses.update(epoch_loss.item(), inputs.size(0))
            top1.update(pred.item(), inputs.size(0))

        valid_loss = losses.avg
        valid_ps1 = top1.avg
        print('valid_loss : ', valid_loss)
        print('valid_ps1 : ', valid_ps1)


        return losses.avg, top1.avg

    
    def predict(self, input_data):

        self.eval()

        input_tensor = torch.from_numpy(input_data).float()
        input_tensor = input_tensor.view((1,-1))

        with torch.no_grad():

            if torch.cuda.is_available():

                input_tensor = input_tensor.cuda()

        output = self.forward(input_tensor)
        pred = torch.argmax(output)

        return pred
        

    def compile(self, optimizer=None, loss=None):        
        if not optimizer is None:
            self.network_optimizer = optimizer

        if not loss is None:
            self.network_loss = loss


class CustomMLP(NetworkBase):
    def __init__(
        self,
        network_config,
        input_shape,
        mode = None,
        saved_model = '',
        *args, **kargs):

        super(CustomMLP, self).__init__(
            network_config,            
            input_shape,
            *args, **kargs)

        self.model = nn.Sequential(
            nn.Linear(input_shape[0], 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Dropout(p=0.2),
            nn.Linear(1024, 3, bias=True),
            nn.BatchNorm1d(3),
            nn.Softmax()
        )

    def forward(self, inputs):

        output = self.model(inputs)

        return output


class ResultsLog():

    def __init__(self, path=''):

        self.path = path
        self.results = None
    
    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)
    def save(self):
        self.results.to_csv(self.path, index=False, index_label=False)


class AverageMeter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':

    ## data read
    data = pd.read_csv("iris.csv").rename(columns={'variety':'species'})

    save_path = 'iris_result'
    os.makedirs(save_path, exist_ok=True)


    ## data analysis
    data_analysis(data, 'iris_result')


    ## data pre-processing
    tr_dataset, va_dataset, index_to_label_dict = data_preprocessing()
    print('index_to_label_dict : ', index_to_label_dict)


    ## set network configuration 
    network_config = SetNetworkConfig(user_defined_name='iris')
    network_config.classes = len(index_to_label_dict)
    print('network_config : ', network_config())
    print()

    ## dataset setting
    train_dataset = Problem4Dataset(tr_dataset)
    valid_dataset = Problem4Dataset(va_dataset)    

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=network_config.batch_size,
        shuffle=True, num_workers=1)
    
    validloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=network_config.batch_size,
        shuffle=False, num_workers=1)


    ### model train
    model = CustomMLP(network_config, input_shape=tr_dataset[0][0].shape)
    print(model)
    model.to(device)
    summary(model, tr_dataset[0][0].shape)
    
    model.compile(
        optimizer=torch.optim.Adam(
            model.parameters(), 
            lr=network_config.initial_lr),
        loss=torch.nn.CrossEntropyLoss())

    scheduler = torch.optim.lr_scheduler.StepLR(
        model.network_optimizer,
        step_size=10, 
        gamma=network_config.var_decay)

    model.fit(trainloader, 
              valid_loader=validloader,
              verbose=1, 
              lr_scheduler=scheduler,
              save_path=save_path)


    ### model load and eval
    checkpoint = torch.load(os.path.join(save_path, 'model_best.pth'))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict']) 
    model.network_loss = checkpoint['loss_func']
    print(model)
    model.to(device)
    summary(model, va_dataset[0][0].shape)

    model.evaluate(validloader)


    ### predict data
    va_x_data = va_dataset[0]

    # predict batch 1 data
    for i, input_data in enumerate(va_x_data):
        
        result = model.predict(input_data)

        pred_label = index_to_label_dict[result.item()]
        print('pred label : ', pred_label)

