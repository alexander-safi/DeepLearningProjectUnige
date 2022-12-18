
# Project 1 – Classification, weight sharing, auxiliary losses

## Quick summary
The objective of this project is to test different architectures to compare two digits visible in a two-channel image. It aims at showing in particular the impact of weight sharing, and of the use of an auxiliary loss to help the training of the main objective.
It should be implemented with PyTorch only code, in particular without using other external libraries such as scikit-learn or numpy.

## How to run the code
The code for project 1 is dispatched between the two files *miniProject1.py* and *miniProject1Modules.py*. The latter contains the classes needed for the first one to run. To start the training and writing of the data, one can simply run *miniProject1.py*. Different parameters can be set in the file, such as number of epochs used or batch size. The code will then go through the 8 possibles configurations 10 times, and write their corresponding score on the test set in a file named *output.txt*.




# Project 2 – Mini deep-learning framework
## Summary
The objective of this project is to design a mini “deep learning framework” using only pytorch’s tensor operations and the standard math library, hence in particular without using autograd or the neural-network modules.
## How to run the code 
The code follows the instructions from the projects sheet. There are also two files, *test.py* that can be run to train the network and display the results on the test and train set and *project2.py* that contains the different classes such as Linear, Sequential, ReLU etc. that are needed for *test.py* to run. The training is done for 150 epochs with a learning rate of 0.001.
