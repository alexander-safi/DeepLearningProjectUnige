from dlc_practical_prologue import generate_pair_sets
from matplotlib import pyplot as plt
from miniProject1Modules import CNN, BCMLP, MLP, AuxMLP
import torch
import torch.nn as nn
import torch.optim as optim
import csv

def init_weights(m):
    """
    This function is used to randomly initialize the weights of the CNNs and the MLPS.
    Fully connected layer's weights are filled with random uniform values between 0 and 0.1 and conv2 weights are filled with random normal
    values of mean 0 and std 0.01
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, std = 0.01)
        m.bias.data.normal_(mean = 0, std = 0.01)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight, std = 0.01)

def train_model(model = "cnn",weight_sharing = False, aux_loss = False, num_epochs = 25):

    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

    bcmlp = BCMLP()
    #bcmlp.apply(init_weights)
    if(model == 'cnn'):
        #If we are using the CNN model, we need to reshape the input images so that they have one channel for the convolutional layers
        train_input = train_input.view(-1,2, 1,14,14).float()
        test_input = test_input.view(-1,2, 1,14,14).float()

        #If we use weight sharing, we only need to define one instance of our CNN
        if(weight_sharing):
            main_model = CNN()
            #main_model.apply(init_weights)
            params = list(main_model.parameters()) + list(bcmlp.parameters())
        #If we are not using weight sharing, we define two instances of the CNN model (the weights will thus not be the same for model1 and model2)
        elif(not weight_sharing):
            model_1 = CNN()
            model_2 = CNN()
            #model_1.apply(init_weights)
            #model_2.apply(init_weights)
            params = list(model_1.parameters()) + list(model_2.parameters()) + list(bcmlp.parameters())

    elif(model == 'mlp'):
        #If we use weight sharing, we only need to define one instance of our MLP
        if(weight_sharing):
            main_model = MLP()
            #main_model.apply(init_weights)
            params = list(main_model.parameters()) + list(bcmlp.parameters())
        #If we are not using weight sharing, we define two instances of the MLP model (the weights will thus not be the same for model1 and model2)
        elif(not weight_sharing):
            model_1 = MLP()
            model_2 = MLP()
            #model_1.apply(init_weights)
            #model_2.apply(init_weights)
            params = list(model_1.parameters()) + list(model_2.parameters()) + list(bcmlp.parameters())
    #If we are using auxiliary loss, we also need to initialize the Aux MLP and the criterion for the auxliary task
    if(aux_loss):
        train_classes = train_classes.view(-1,2,1)
        lambda_aux = 0.5
        #we use cross entropy as our criterion for the aux loss (cross entropy contains a softmax so no need to add one)
        aux_criterion = nn.CrossEntropyLoss()
        aux_MLP = AuxMLP()
        #aux_MLP.apply(init_weights)

        params += list(aux_MLP.parameters())
    
    
   
    #We use binary cross entropy as our main criterion
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.SGD(params, lr=0.001)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i in torch.randperm(len(train_input)):

            # zero the parameter gradients
            optimizer.zero_grad()

            # we pass the 2 images through the forwad, which results in two tensors of size 16
            if(weight_sharing):
                x1 = main_model.forward(train_input[i,0])
                x2 = main_model.forward(train_input[i,1])
            elif(not weight_sharing):
                x1 = model_1.forward(train_input[i,0])
                x2 = model_2.forward(train_input[i,1])
            
            if(aux_loss):
                #We pass the images through an auxiliary MLP to compute the auxiliary loss
                aux_x1 = aux_MLP.forward(x1)
                aux_x2 = aux_MLP.forward(x2)
            
            #We concatenate the two tensors x1 and x2 to pass them through the final MLP 
            concat_data = torch.concat((x1,x2), 1)
            #This mlp reduces the data to a tensor of size 1, that is then passed through a sigmoid
            output = bcmlp(concat_data)[0][0]#.to(torch.float32)
            #we compute the loss with the binary classifier
            loss = criterion(output, train_target[i].to(torch.float32))

            if(aux_loss):
                loss2 = aux_criterion(aux_x1, train_classes[i,0])
                loss3 = aux_criterion(aux_x2, train_classes[i,1])
                loss +=   lambda_aux*(loss2 + loss3)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        #print('Loss after ' + str(epoch + 1) + ' iterations : ', running_loss)
        running_loss = 0

    print('Finished Training')
    if(weight_sharing and not aux_loss):
        return main_model, main_model, bcmlp, None
    elif(weight_sharing and aux_loss):
        return main_model, main_model, bcmlp, aux_MLP
    elif(not weight_sharing and not aux_loss):
        return model_1, model_2, bcmlp, None
    elif(not weight_sharing and aux_loss):
        return model_1, model_2, bcmlp, aux_MLP
    

def predict_output(sample, target, trained_model1, trained_model2, trained_bcmlp, trained_aux ):
    t1 = trained_model1.forward(sample[0])
    t2 = trained_model2.forward(sample[1])

    concat_data = torch.concat((t1,t2), 1)

    res = trained_bcmlp.forward(concat_data)[0][0]
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
    

#We write the results in the file 'output.txt'

with open('output.txt', 'w') as f:
    writer = csv.writer(f)
    #We write the columns
    f.write('Model,WeightSharing,AuxLoss,RunNb,Score\n')
    #Loop over the two architectures
    for chosen_model in ['cnn', 'mlp']:
        #Loop over the two possibilities for weight sharing : with and without
        for weight_sharing_active in [True, False]:
            #Loop over the two possibilities for auxiliary loss : with or without
            for aux_loss_active in [True, False]:

                for k in range(10):

                    trained_model1, trained_model2, trained_bcmlp, trained_aux = train_model(model = chosen_model,weight_sharing = weight_sharing_active, aux_loss = aux_loss_active, num_epochs = 25)

                    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

                    if(chosen_model == 'cnn'):
                        test_input = test_input.view(-1,2, 1,14,14).float()
                    n_correct = 0
                    n_false = 0
                    for j in range(len(test_input)):
                        if(predict_output(test_input[j], test_target[j], trained_model1, trained_model2, trained_bcmlp, trained_aux)):
                            n_correct += 1
                        else:
                            n_false += 1


                    score = n_correct/(n_correct+n_false)
                    print(str(100*n_correct/(n_correct+n_false)) + '% of correct answers', k+1, 'th run')
                    writer.writerow([chosen_model,str(weight_sharing_active),str(aux_loss_active),str(k),str(score)])