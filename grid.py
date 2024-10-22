import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import torch
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold #import k fold

class MinstDataset(data.Dataset):
    def __init__(self):
        train = pd.read_csv("grid_data.csv")
        #train_labels = train['stabf'].values
        train_labels = train['stab'].values
        train = train.drop("stab",axis=1)
        train = train.drop("stabf",axis=1)
        train = train.values
        self.datalist = train
        self.labels = train_labels
    def __getitem__(self, index):
        return torch.Tensor(self.datalist[index].astype(float)), self.labels[index]
    def __len__(self):
        return self.datalist.shape[0]

class Model(nn.Module):

    def __init__(self, num_numerical_cols, output_size, layers, p=0.5):
        super().__init__()
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        input_size = num_numerical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_numerical):
        x_numerical = self.batch_norm_num(x_numerical)
        x = self.layers(x_numerical)
        return x

dataset = pd.read_csv(r'grid_data.csv')
numerical_columns = ['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2','g3', 'g4']
named_output = ['stabf']
numerical_output = ['stab']


numerical_data = np.stack([dataset[col].values for col in numerical_columns], 1)
numerical_data = torch.tensor(numerical_data, dtype=torch.float)
numerical_output = torch.tensor(dataset[numerical_output].values).flatten()

total_records = 10000
test_records = int(total_records * .2)

#Validate healthy data split with K Fold Cross Validation
kf = KFold(n_splits=5)

for train_index, test_index in kf.split(numerical_data):
    #train the model
    model = Model(numerical_data.shape[1], 1, [50,25,10], 0.05)
    numerical_train_data = numerical_data[train_index]
    numerical_test_data = numerical_data[test_index]
    train_outputs = numerical_output[train_index]
    test_outputs = numerical_output[test_index]
    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    epochs = 300
    aggregated_losses = []
    for i in range(epochs):
        i += 1
        y_pred = model(numerical_train_data)
        train_outputs = train_outputs.view(-1,1)
        single_loss = loss_function(y_pred, train_outputs.float())
        aggregated_losses.append(single_loss)

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        optimizer.zero_grad()
        single_loss.backward()
        optimizer.step()

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    with torch.no_grad():
        y_val = model(numerical_test_data)
        test_outputs = test_outputs.view(-1,1)
        loss = loss_function(y_val, test_outputs.float())
    print(f'Loss: {loss:.8f}')

    bin_y_val = [None] * 2000
    named_y = [None] * 2000
    for z in range(2000):
        if y_val[z] > 0:
            bin_y_val[z] = 1
            named_y[z] = 'unstable'
        else:
            bin_y_val[z] = 0
            named_y[z] = 'stable'
    to = [None] * 2000
    nto = [None] * 2000
    for z in range(2000):
        if test_outputs[z] > 0:
            to[z] = 1
            nto[z] = 'unstable'
        else:
            to[z] = 0
            nto[z] = 'stable'

    print(confusion_matrix(to,bin_y_val))
    print(classification_report(to,bin_y_val))
    print(accuracy_score(to,bin_y_val))
