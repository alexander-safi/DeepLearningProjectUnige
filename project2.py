from torch import empty
import math
from torch import set_grad_enabled

set_grad_enabled(False)

class Linear():
    """
    This class implements a dense layer in the same style as the Linear class from pytorch
    """
    def __init__(self, n_in, n_out, bias =  True):
        #We first initialize all the parameters of the class
        self.n_in = n_in
        self.n_out = n_out
        self.output = []
        self.input = []
        self.dl_dw = empty((n_out, n_in))
        self.dl_db = empty(n_out)
        self.grad = empty(1)
        #From pytorch documentation, we know that the weights are initialized randomly
        #using uniform distribution between -sqrt(k) and sqrt(k) where k = 1/n_in
        k = 1/n_in
        self.weights = empty((n_out, n_in)).uniform_(-math.sqrt(k), math.sqrt(k))
        self.bias = empty(n_out).uniform_(-math.sqrt(k), math.sqrt(k))

        
        return
    def __call__(self, X):
        """
        This is the forward call of the class
        """
        self.input = X
        self.output = (X @ self.weights.T + self.bias).squeeze_()
        
        return self.output
    def param(self):
        return [self.weights , self.bias, self]
    
    def backward(self, gradientwrtoutput, next_layer):
        """
        Computes the gradient of this layer
        Inputs :
        gradientwrtoutput : the gradient value of layer l+1 (considering this is layer l)
        next_layer : the class object of the next layer, needs to be an activation function (relu or tanh)

        """
        self.grad = next_layer.gradient(self.output) * gradientwrtoutput

        self.dl_dw += self.grad.view(-1,1).mm(self.input.view(1,-1))
        self.dl_db += self.grad
        return self.grad
    def update_weights(self, learning_rate):
        """
        Updates the weights according to the learning rate
        """
        self.weights -= learning_rate*self.dl_dw
        self.bias -= learning_rate*self.dl_db
    def zero_grad(self):
        """
        This function resets all the gradients to zero
        """
        self.dl_dw = self.dl_dw.zero_()
        self.dl_db = self.dl_db.zero_()
        self.grad = self.grad.zero_()

class relu():
    """
    Relu class that allows to perform the Relu activation function
    """

    def __call__(self,X):
        """
        This is the forward
        """

        return (X>0)*X
    def gradient(self, X):
        """
        Computes the derivative of the relu
        """
        return (X>0).float()
    def param(self):
        return []
    def backward(self, gradientwrtoutput, next_layer):
        """
        Computes the backward of the layer.
        gradientwrtouput: gradient of the layer l+1 (considering this is layer l)
        next_layer : object of the next layer. Needs to be of type Linear
        """
        return next_layer.weights.T@gradientwrtoutput
    def zero_grad(self):
        """
        This function does not do anything but is defined because it gets called when we reset all the gradients
        """
        return
    def update_weights(self, learning_rate):
        """
        This function does not do anything but is defined because it gets called when we reset update all the weights
        """
        return

class tanh():
    """
    This is the tanh class that defines the tanh activation function
    """

    def __call__(self, X):
        """
        This is the forward. It returns tanh(X)
        """

        return 2/(1+ (-2*X).exp()) -1
    def gradient(self, X):
        """
        This returns the derivative of tanh of X
        """
        return 1-(X.tanh())**2
    def param(self):
        return []
    def backward(self, gradientwrtoutput, next_layer):
        """
        Computes the backward of the layer.
        gradientwrtouput: gradient of the layer l+1 (considering this is layer l)
        next_layer : object of the next layer. Needs to be of type Linear
        """
        return next_layer.weights.T@gradientwrtoutput
    def zero_grad(self):
        """
        This function does not do anything but is defined because it gets called when we reset all the gradients
        """
        return
    def update_weights(self, learning_rate):
        """
        This function does not do anything but is defined because it gets called when we update all the weights
        """
        return

class MSE():
    """
    This is the class that defines the mean squared error 
    """
    def __call__(self, X, y):
        """
        This is the forward of the class. It returns MSE(X,y) where X is the output of the model and y the target
        """
        return ((y-X)**2).sum()/len(y)

    def gradient(self, result, label):
        """
        Computes the gradient of the loss
        """
        return 2*(result-label)/len(label)

class Sequential():
    """
    This is the sequential class. It allows to create a sequence of layers to create the network.
    """
    def __init__(self, *layers):
        #We create a list containing the layers and a list containing the parameters
        self.layers = []
        self.parameters = []
        
        #For each layer, we add the object to the layers list and the parameters to the parameters list
        for layer in layers:
            self.layers.append(layer)
            self.parameters.append(layer.param())
            
    def __call__(self, X):
        """
        The foward iterativeyl calls the forward of each layer on the output of the previous layer
        """
        for i in range(len(self.layers)):
            X = self.layers[i](X)
        return X
    def backward(self, dloss):
        """
        The backward takes as input the loss of the final layer dloss
        """
        #We then call the backward of all the layers iteratively, starting from the last layer
        for layer_index in range(len(self.layers)-1)[::-1]:
            layer = self.layers[layer_index]
            #We update the value in dloss at each layer and pass it as argument for the next layer in the list
            dloss = layer.backward(dloss, self.layers[layer_index + 1]) 
        return
    def update_weights(self):
        """
        This function allows to update the weights of all the parameters in the layers
        """
        #We loop through all the layers
        for layer in self.layers:
            #and we call their own update_weights function
            layer.update_weights(self.learning_rate)
        return
    def param(self):
        #This function simply return the list of all the parameters
        return self.parameters
    def zero_grad(self):
        """
        This function allows to reset the gradients in all the layers
        """
        #Loop thrgouh all the layers and call their own zero_grad()
        for layer in self.layers:
            layer.zero_grad()
    def set_learning_rate(self, learning_rate):
        #This function allows to set the learning rate of the network
        self.learning_rate = learning_rate