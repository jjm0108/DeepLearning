import pandas as pd
df = pd.read_csv('DeepLearning/Pytorch/PV_Elec_Gas3.csv').rename(columns={'date':'timestamp'}).set_index('timestamp')
train_set = df[:'31/10/2018']
valid_set = df['1/11/2018':'18/11/2019']
print("*"*50)
print(train_set)

df_n = pd.read_csv('DeepLearning/Pytorch/PV_Elec_Gas3.csv').rename(columns={'date':'timestamp'})#.set_index('timestamp')
train_set_split_ratio = 0.78
valid_set_split_ratio = 0.12
train_set_split = int(len(df)*train_set_split_ratio)
valid_set_split = int(len(df)*(train_set_split_ratio + valid_set_split_ratio))

train_set_n = df_n[:train_set_split]
valid_set_n = df_n[train_set_split:valid_set_split]
print("*"*50)
print(train_set_n)