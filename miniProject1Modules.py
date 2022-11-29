
import torch
import torch.nn as nn
import torch.nn.functional as functional


class CNN(nn.Module):
    """
    This is the CNN that we use for the weight sharing task
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels=16, kernel_size = 4, stride = 1, padding = 0 )
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size= 4, stride = 1, padding = 0)
        self.fc = nn.Linear(32, 16)
        
    def forward(self, x):
        #The forward consists of two convolutional layers
        #And we apply relu and maxpool inbetween
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(-1, 32*1*1)
        #We finish with a fully connected layer and the final output is of size 16
        x = self.fc(x)
        return x

class BCMLP(nn.Module):
    """
    This is the final MLP that we use at the end to predict the binary result.
    We use this same mlp in all different cases (weight sharing / no weight sharing),
    with / without aux loss
    """
    def __init__(self):
        super(BCMLP, self).__init__()
        self.fc1 = nn.Linear(32, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50,1)

    def forward(self, x):
        #This mlp simply consists in three fully connected layers
        #We apply relu after layer 1 and 2
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        #The final output is of shape 1 and represents the boolean value for the main task
        x = self.fc3(x)
        return x
    
class CNN2(nn.Module):
    """
    This is the CNN that we use for the weight sharing task. It is essentially the same as the one for the task without weight sharing, except the first layer
    takes 2 channels as input (each of the channel is one of the images of the pair)
    """
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 2, out_channels=32, kernel_size = 4, stride = 1, padding = 0 )
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size= 4, stride = 1, padding = 0)
        self.fc = nn.Linear(64, 32)
        
    def forward(self, x):
        #The forward consists of two convolutional layers
        #And we apply relu and maxpool inbetween
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(-1, 64*1*1)
        #We finish with a fully connected layer and the final output is of size 32
        x = self.fc(x)
        return x

class MLP(nn.Module):
    """
    This is the MLP that we use for the weight sharing task. It takes as input a single image, and output a tensor of size 16 than can then be passed
    through the secondary MLP (needs to be concatenated with another output of size 16 first).
    """
    def __init__(self):
        super(MLP,self).__init__()

        self.fc1 = nn.Linear(14*14, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,16)
        
    def forward(self,x):
        # we first flatten the image
        x = x.view(-1,14*14)
        # we then take relu of first fully connected layer
        x = functional.relu(self.fc1(x))
         # and then relu of second fully connected layer
        x = functional.relu(self.fc2(x))
        # we finally return the result of the third layer which is of shape 16
        x = self.fc3(x)
        return x

class MLP2(nn.Module):
    """
    This is the MLP that we use for the no weight sharing task. It takes as input the pair of images, and outputs a tensor of size 32 that can be directly passed
    as input to the secondary mlp
    """
    def __init__(self):
        super(MLP2,self).__init__()

        self.fc1 = nn.Linear(2*14*14, 2*256)
        self.fc2 = nn.Linear(2*256,2*256)
        self.fc3 = nn.Linear(2*256,2*16)
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,14*14*2)
        # we add relu of first fully connected layer
        x = functional.relu(self.fc1(x))
         # add hidden layer, with relu activation function
        x = functional.relu(self.fc2(x))
        # add output layer
        x = self.fc3(x)
        return x
    
class AuxMLP(nn.Module):
    """
    This is the MLP that is used for the auxiliary loss computation. It takes as input 
    """
    def __init__(self):
        super(AuxMLP,self).__init__()

        self.fc1 = nn.Linear(16, 128)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(128,128)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(128,10)
    def forward(self, x):
        # we add relu of first fully connected layer
        x = functional.relu(self.fc1(x))
         # add hidden layer, with relu activation function
        x = functional.relu(self.fc2(x))
        # add output layer
        x = self.fc3(x)
        return x