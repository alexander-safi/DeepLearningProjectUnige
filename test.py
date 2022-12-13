from math import pi
from project2 import Sequential, relu, Linear, tanh, MSE
from torch import empty
from torch import set_grad_enabled

set_grad_enabled(False)


train_set = empty((1000,2)).uniform_(0,1)
test_set = empty((1000,2)).uniform_(0,1)

train_label = ((train_set[:,0]-0.5)**2 + (train_set[:,1]-0.5)**2) <= 1/(2*pi)
test_label = ((test_set[:,0]-0.5)**2 + (test_set[:,1]-0.5)**2) <= 1/(2*pi)

train_label = train_label.float()
test_label = test_label.float()

#We create the network using our custom Sequential class
network = Sequential(Linear(2,25), relu(),
                            Linear(25,25), relu(),
                            Linear(25,25), relu(),
                            Linear(25,25), relu(),
                            Linear(25,1),tanh())

#We set the learning rate of our network using the set_learning_rate method
network.set_learning_rate(0.001)
#We create our loss object
loss = MSE()

#We train the model for 150 epochs
for i in range(150):
    
    running_loss = 0
    #We run through all the 1000 points
    for j in range(len(train_set)):
        #We set the gradients in the layers to 0
        network.zero_grad()
        #We compute the result with the forward of our network
        result = network(train_set[j])
        #We get the running loss
        running_loss += loss(result, train_label[j].view(-1))
        #We compute the derivative of the loss for the last layer
        dloss = loss.gradient(result, train_label[j].view(-1))
        
        #We compute the back propagation starting from the last layer
        network.backward(dloss)
        #and we update all the weights
        network.update_weights()
    if(i%10 == 0):
        print('iter ',i,' loss : ',running_loss.item())

#We check the accuracy of our trained model
correct = 0

for i in range(len(test_set)):
    result = network(test_set[i])
    result_train = network(train_set[i])
    if(result.item() > 0.5 and test_label[i] == 1):
        correct += 1
    elif(result.item() <= 0.5 and test_label[i] == 0):
        correct += 1
    else:
        pass
print('Accuracy on test set: ' ,100*correct/len(test_set), "%")

#we do the same for the train set
correct_train = 0
for i in range(len(train_set)):
    result_train = network(train_set[i])
    if(result_train.item() > 0.5 and train_label[i] == 1):
        correct_train += 1
    elif(result_train.item() <= 0.5 and train_label[i] == 0):
        correct_train += 1
    else:
        pass
print('Accuracy on train set: ' ,100*correct_train/len(train_set), "%")