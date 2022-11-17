from dlc_practical_prologue import generate_pair_sets
from matplotlib import pyplot as plt

train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

train_input = train_input.view(-1,2, 1,14,14).float()
test_input = test_input.view(-1,2, 1,14,14).float()


import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels=16, kernel_size = 4, stride = 1, padding = 0 )
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size= 4, stride = 1, padding = 0)
        self.fc = nn.Linear(32, 16)
        
    def forward(self, x):
        #print(x.shape)
        x = self.pool(functional.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(functional.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 32*1*1)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x

class CNN3(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels=16, kernel_size = 4, stride = 1, padding = 0 )
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size= 4, stride = 1, padding = 0)
        self.fc = nn.Linear(32, 16)
        
    def forward(self, x):
        #print(x.shape)
        x = self.pool(functional.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(functional.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 32*1*1)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x
    

class CNN2(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = CNN()
        
    def forward(self, x):
        #print(x.shape)
        x = self.pool(functional.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(functional.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 32*1*1)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x
    

cnn = CNN()

mlp = nn.Sequential(
    nn.Linear(32, 200),
    nn.ReLU(),
    nn.Linear(200, 50),
    nn.ReLU(),
    nn.Linear(50,1)
)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001)

for epoch in range(25):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_input):
        # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        x1 = cnn.forward(train_input[i,0])
        x2 = cnn.forward(train_input[i,1])

        concat_data = torch.concat((x1,x2), 1)

        output = mlp(concat_data)[0][0]#.to(torch.float32)
        loss = criterion(output, train_target[i].to(torch.float32))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 900 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')


def predict(sample, target):
    t1 = cnn.forward(sample[0])
    t2 = cnn.forward(sample[1])

    concat_data = torch.concat((t1,t2), 1)

    res = mlp(concat_data)[0][0]
    if torch.sigmoid(res) > 0.5 and target.item() == 1:
        #print("Prediction : First number <= second one, truth : ", target)
        #print((torch.sigmoid(res) > 0.5 ) == target)
        return True
    elif torch.sigmoid(res) <= 0.5 and target.item() == 0:
        #print("Prediction : First number > second one, truth : ", target)
        #print((torch.sigmoid(res) <= 0.5 ) == target)
        return True
    else:
        return False
    
n_correct = 0
n_false = 0
for j in range(len(test_input)):
    if(predict(test_input[j], test_target[j])):
        n_correct += 1
    else:
        n_false += 1

print(str(100*n_correct/(n_correct+n_false)) + '% of correct answers')