import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch.optim as optim

filename = 'events.csv'
events = pd.read_csv(filename)
shots = events[(events.event_type==1)]

# only the last 6 columns are relevant for shots
shots_prediction = shots.iloc[:,-6:]
shot_data = pd.get_dummies(shots_prediction, columns=['location', 'bodypart','assist_method', 'situation'])
shot_data.columns = ['is_goal', 'fast_break', 'loc_centre_box', 'loc_diff_angle_lr', 'diff_angle_left', 
                    'diff_angle_right', 'left_side_box', 'left_side_6ybox', 'right_side_box', 'right_side_6ybox', 
                    'close_range', 'penalty', 'outside_box', 'long_range', 'more_35y', 'more_40y', 'not_recorded', 
                    'right_foot', 'left_foot', 'header', 'no_assist', 'assist_pass', 'assist_cross', 'assist_header', 
                    'assist_through_ball', 'open_play', 'set_piece', 'corner', 'free_kick']


shot_chars = shot_data.iloc[:,1:]
shot_results = shot_data.iloc[:,0]
X = torch.FloatTensor(shot_chars.values)
y = torch.FloatTensor(shot_results.values)



VAL_PCT = 0.2
val_size = int(len(X)*VAL_PCT)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, input_size // 2)
        self.fc3 = nn.Linear(input_size // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)


net = Net(len(X[0]))

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.BCELoss()

BATCH_SIZE = 100
EPOCHS = 1000

# training step
for epoch in tqdm(range(EPOCHS)):
    optimizer.zero_grad()
    for i in range(0, len(train_X), BATCH_SIZE):
        batch_X = train_X[i:i+BATCH_SIZE]
        batch_y = train_y[i:i+BATCH_SIZE]
        net.zero_grad()
        output = net(batch_X).view(-1)
        loss = loss_function(output, batch_y)
        loss.backward() # stochastic gradient descent
        optimizer.step() # updates the weights
    if epoch % 100 == 0:
        print(loss)


# testing step
correct = 0
total = 0

with torch.no_grad():
    for i in range(len(test_X)):
        output = net(test_X[i])[0]
        if output < 0.5:
            output = 0
        else:
            output = 1
        if output == test_y[i]:
            correct += 1
        total += 1

print("Accuracy: ", round(correct/total, 3))