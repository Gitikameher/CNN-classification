#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
class M_Help:

    def __init__(self):
        self.epsilon=1e-20
    
    def accuracy(self,outputs, labels):
        o=(outputs>=0.5).float()
        return torch.sum(o==labels.float(),0).float()/labels.float().shape[0]

    def true_positive(self,outputs, labels):
        o=(outputs>=0.5).float()
        return torch.sum(o+labels.float() ==2.,0)
    
    def true_negative(self,outputs, labels):
        o=(outputs>=0.5).float()
        return torch.sum(o+labels.float() ==0.,0)

    def false_positive(self,outputs, labels):
        o=(outputs>=0.5).float()
        return torch.sum((labels.float()-o)<0,0)

    def false_negative(self,outputs, labels):
        o=(outputs>=0.5).float()
        return torch.sum((o-labels.float())<0,0)

    def precision(self,outputs, labels):
        return torch.div(self.true_positive(outputs, labels).float()+self.epsilon,(self.true_positive(outputs, labels).float()+self.false_positive(outputs, labels).float()+self.epsilon))

    def recall(self,outputs, labels):
        return torch.div(self.true_positive(outputs, labels).float(),(self.true_positive(outputs, labels).float()+self.false_negative(outputs, labels).float()+self.epsilon))

    def BCR(self,outputs, labels):
        return (self.precision(outputs, labels)+self.recall(outputs, labels))/2


# In[7]:


################################################################################
# CSE 253: Programming Assignment 3
# Winter 2019
# Code author: Ambareesh Farheen
#
# Filename: baseline_cnn.py
#
# Description:
#
# This file contains the starter code for the baseline architecture you will use
# to get a little practice with PyTorch and compare the results of with your
# improved architecture.
#
# Be sure to fill in the code in the areas marked #TODO.
################################################################################


# PyTorch and neural network imports
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
from xray_dataloader import ChestXrayDataset, create_split_loaders

import matplotlib.pyplot as plt
import numpy as np
import os
import math


class CNN_1(nn.Module):
    """ A basic convolutional neural network model for baseline comparison.

    Consists of three Conv2d layers, followed by one 3x3 max-pooling layer,
    and 2 fully-connected (FC) layers:

    conv1 -> conv2 -> conv3 -> maxpool -> fc1 -> fc2 (outputs)

    Make note:
    - Inputs are expected to be grayscale images (how many channels does this imply?)
    - The Conv2d layer uses a stride of 1 and 0 padding by default
    """

    def __init__(self):
        n = 512
        super(CNN_1, self).__init__()

        # conv1: 1 input channel, 12 output channels, [8x8] kernel size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=8)
        self.conv1_normed = nn.BatchNorm2d(4)
        torch_init.xavier_normal_(self.conv1.weight)

        # conv2: X input channels, 10 output channels, [8x8] kernel, initialization: xavier
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=6)
        self.conv2_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv2.weight)

        # TODO: Apply max-pooling with a [3x3] kernel using tiling (*NO SLIDING WINDOW*)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=1)

        # conv3: X input channels, 8 output channels, [6x6] kernel, initialization: xavier
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=8)
        self.conv3_normed = nn.BatchNorm2d(12)
        torch_init.xavier_normal_(self.conv3.weight)

        # conv4: X input channels, 8 output channels, [6x6] kernel, initialization: xavier
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=6)
        self.conv4_normed = nn.BatchNorm2d(16)
        torch_init.xavier_normal_(self.conv4.weight)

        # TODO: Apply max-pooling with a [3x3] kernel using tiling (*NO SLIDING WINDOW*)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        max_pool_output_count = (math.floor((n - 30) / 3) + 1) ** 2
        max_pool_channel_count = 16

        # Define 3 fully connected layers:
        in_feature_fc1_count = max_pool_channel_count * (max_pool_output_count)

        self.fc1 = nn.Linear(in_features=in_feature_fc1_count, out_features=256)
        self.fc1_normed = nn.BatchNorm1d(256)
        torch_init.xavier_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc2_normed = nn.BatchNorm1d(128)
        torch_init.xavier_normal_(self.fc2.weight)

        # TODO: Output layer: what should out_features be?
        self.fc3 = nn.Linear(in_features=128, out_features=14)

    def forward(self, batch):
        """Pass the batch of images through each layer of the network, applying
        non-linearities after each layer.

        Note that this function *needs* to be called "forward" for PyTorch to
        automagically perform the forward pass.

        Params:
        -------
        - batch: (Tensor) An input batch of images

        Returns:
        --------
        - logits: (Variable) The output of the network
        """

        # Apply first convolution, followed by ReLU non-linearity;
        # use batch-normalization on its outputs
        batch = func.relu(self.conv1_normed(self.conv1(batch)))
        batch = func.relu(self.conv2_normed(self.conv2(batch)))
        batch = self.pool1(batch)

        batch = func.relu(self.conv3_normed(self.conv3(batch)))
        batch = func.relu(self.conv4_normed(self.conv4(batch)))
        batch = self.pool2(batch)

        # Reshape the output of the conv3 to pass to fully-connected layer
        batch = batch.view(-1, self.num_flat_features(batch))
        batch = func.relu(self.fc1(batch))

        batch = func.relu(self.fc2(batch))

        batch = self.fc3(batch)
        # batch = torch.sigmoid(batch)  Because we are shifting to BCE with Logits Loss

        return batch

    def num_flat_features(self, inputs):
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features

    def loss_function(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def using_pytorch_weight_loss(self):
        print("###_Version2")
        
        weights = np.array([8.699801021,
                            39.38904899,
                            7.419313659,
                            4.635870112,
                            18.39121411,
                            16.70968251,
                            77.35080363,
                            20.14673708,
                            23.02399829,
                            47.68432479,
                            43.56279809,
                            65.50059312,
                            32.1225997,
                            492.9207048], dtype=np.float32)

#         weights = np.array([16,
#                             70,
#                             15,
#                             4.635870112,
#                             36,
#                             16.70968251,
#                             90,
#                             30,
#                             23.02399829,
#                             47.68432479,
#                             65,
#                             90,
#                             32.1225997,
#                             492.9207048], dtype=np.float32)
        
    
        w_n = torch.tensor( [1.114945158,
                            1.025387767,
                            1.134783357,
                            1.215709236,
                            1.054373789,
                            1.059845542,
                            1.012928114,
                            1.049635829,
                            1.043432943,
                            1.020971252,
                            1.022955367,
                            1.015267037,
                            1.031130731,
                            1.002028724,
                            ])
        
        
        
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            computing_device = torch.device("cuda")
            extras = {"num_workers": 1, "pin_memory": True}
            print("CUDA is supported")
        else:  # Otherwise, train on the CPU
            computing_device = torch.device("cpu")
            extras = False
            print("CUDA NOT supported")

        weights /= 1  #to get better recall 
        weights = torch.Tensor(weights)
        weights.required_grad = False
        weights = weights.to(computing_device)
        
        w_n = w_n.to(computing_device)

        
        criterion = nn.BCEWithLogitsLoss(pos_weight = weights, weight = w_n)  #Working, but low accuracy
   
        return criterion    


# In[8]:


###################################
###################################
###################################

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import metric_helper

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
from xray_dataloader import ChestXrayDataset, create_split_loaders

import matplotlib.pyplot as plt
import numpy as np
import os
import math

# Setup: initialize the hyperparameters/variables
num_epochs = 5  # Number of full passes through the dataset
batch_size = 32 #16  # Number of samples in each minibatch
learning_rate = 0.001
seed = np.random.seed(1)  # Seed the random number generator for reproducibility
p_val = 0.1  # Percent of the overall dataset to reserve for validation
p_test = 0.2  # Percent of the overall dataset to reserve for testing
momentum = 0.9


# TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else:  # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

# Setup the training, validation, and testing dataloaders
train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, transform=transform,
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False,
                                                             extras=extras)

# Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
model = CNN_1()
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

# TODO: Define the loss criterion and instantiate the gradient descent optimizer
# Weights computed after looking at data . Are inversely propotional to their occurence

# TODO - loss criteria are defined in the torch.nn package
# criterion = model.loss_function()
criterion = model.using_pytorch_weight_loss()

# TODO: Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Track the loss across training
total_loss = []
avg_minibatch_loss = []

# Begin training procedure
for epoch in range(num_epochs):

    N = 50
    N_minibatch_loss = 0.0

    # Get the next minibatch of images, labels for training
    for minibatch_count, (images, labels) in enumerate(train_loader, 0): 
        
        
#         if minibatch_count == 100:
#             break
            
        print("mini_batch", minibatch_count)
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, labels = images.to(computing_device), labels.to(computing_device)

        # Zero out the stored gradient (buffer) from the previous iteration
        optimizer.zero_grad()

        # Perform the forward pass through the network and compute the loss
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Automatically compute the gradients and backpropagate the loss through the network
        loss.backward()

        # Update the weights
        optimizer.step()
        
        # Add this iteration's loss to the total_loss
        total_loss.append(loss.item())
        N_minibatch_loss += loss

        if minibatch_count % N == 5: # change to 49
            # Print the loss averaged over the last N mini-batches
            N_minibatch_loss /= N
            print('Epoch %d, average minibatch %d loss: %.3f' % (epoch + 1, minibatch_count + 1, N_minibatch_loss))
            # Add the averaged loss over N minibatches and reset the counter
            avg_minibatch_loss.append(N_minibatch_loss)
            N_minibatch_loss = 0.0

    print("Finished", epoch + 1, "epoch(s) of training")

    # TODO: Implement validation #with torch.no_grad():
    true_positives, true_negatives, false_positives, false_negatives = 0.0, 0.0, 0.0, 0.0
    acc, preci, recall,bcr = 0.0, 0.0,0.0,0.0
    metric_helper = M_Help()
    
    with torch.no_grad():
        for data_val in val_loader:
            images_val, labels_val = data_val
            labels_val = labels_val.type(torch.uint8)
            images_val, labels_val = images_val.to(computing_device), labels_val.to(computing_device)

            outputs_val = torch.sigmoid(model(images_val))
                
            true_positives += metric_helper.true_positive(outputs_val,labels_val)
            true_negatives += metric_helper.true_negative(outputs_val,labels_val)
            false_positives+= metric_helper.false_positive(outputs_val,labels_val)
            false_negatives+= metric_helper.false_negative(outputs_val,labels_val)
            
    epsilon = 1e-10
    print("METRICS")
    print("TP: ", true_positives)
    print("TN: ", true_negatives)
    print("FP: ", false_positives)
    print("FN:", false_negatives)
    print("ACC: ", (true_positives + true_negatives).float() / (true_positives + true_negatives + false_positives+false_negatives).float())
    print("PRECISION: ",(true_positives.float() / (true_positives.float() + false_positives.float() + epsilon).float()))
    print("RECALL: ", (true_positives.float() / (true_positives.float() + false_negatives.float() + epsilon).float()))
    print("BCR: ", (true_positives.float() / (true_positives.float() + false_positives.float() + epsilon).float()) * 0.5 + (
            true_positives.float() / (true_positives.float() + false_negatives.float() + epsilon).float()) * 0.5)        
    ##Val End

print("Training complete after", epoch, "epochs")


# In[ ]:




