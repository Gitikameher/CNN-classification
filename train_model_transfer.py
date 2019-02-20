#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
print(torch.__version__)


# In[2]:


torch.cuda.is_available()


# In[3]:


import warnings

from experiment import *
from experiment import BasicCNN
import weighted_loss as wl
import torchvision.models as models
import metric_helper
import time
import copy

# CUDA_LAUNCH_BLOCKING=1
# Setup: initialize the hyperparameters/variables
num_epochs = 100           # Number of full passes through the dataset
batch_size = 16          # Number of samples in each minibatch
learning_rate = 0.001  
seed = np.random.seed(1) # Seed the random number generator for reproducibility
p_val = 0.1              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing

#TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
#for tranfer learning we have the model create fake rgb channels and normalize as needed
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])


# Check if your system supports CUDA

use_cuda = torch.cuda.is_available()
print(torch.cuda.is_available())
#use_cuda =0

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")
                
                                                                                
# Setup the training, validation, and testing dataloaders
train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, transform=transform, 
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=True, 
                                                             extras=extras)

# Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
model = BasicCNN()
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

# Load the pretrained model from pytorch
vgg16 = models.vgg16(pretrained=True)

print(vgg16.classifier[6].out_features) # 1000 


# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False
    
# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 14)]) # Add our layer with 14 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
vgg16 = vgg16.to(computing_device)
print(vgg16)


# In[4]:


#TODO: Define the loss criterion and instantiate the gradient descent optimizer
#criterion = nn.BCEWithLogitsLoss() #TODO - loss criteria are defined in the torch.nn package
#criterion = wl.weighted_loss_custom(vgg16)
criterion = wl.using_pytorch_weight_loss(vgg16)

#TODO: Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.Adam(vgg16.parameters(), lr=1e-05) #TODO - optimizers are defined in the torch.optim package
#optimizer = optim.SGD(vgg16.parameters(), lr = 5e-04, momentum=0.9)


# In[5]:


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


# In[6]:


# Track the loss across training
total_loss = []
avg_minibatch_loss = []

total_loss_ = []
avg_minibatch_loss_ = []

# Begin training procedure
for epoch in range(num_epochs):
    vgg16.train(True)
    N = 50
    N_minibatch_loss = 0.0

    # Get the next minibatch of images, labels for training
    for minibatch_count, (images, labels) in enumerate(train_loader, 0): 
        if minibatch_count == 500:
            break
            
        print("mini_batch", minibatch_count)
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, labels = images.to(computing_device), labels.to(computing_device)

        # Zero out the stored gradient (buffer) from the previous iteration
        optimizer.zero_grad()

        # Perform the forward pass through the network and compute the loss
        outputs = vgg16(images)

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
    vgg16.train(False)
    vgg16.eval()
    correct_val = 0.0
    total_val = 0.0

    true_positives, true_negatives, false_positives, false_negatives = 0.0, 0.0, 0.0, 0.0
    acc, preci, recall,bcr = 0.0, 0.0,0.0,0.0
    metric_helper = M_Help()
    
    with torch.no_grad():
        for data_val in test_loader:
            images_val, labels_val = data_val
            labels_val = labels_val.type(torch.uint8)
            images_val, labels_val = images_val.to(computing_device), labels_val.to(computing_device)

            outputs_val = torch.sigmoid(vgg16(images_val))
                
            true_positives += metric_helper.true_positive(outputs_val,labels_val)
            true_negatives += metric_helper.true_negative(outputs_val,labels_val)
            false_positives+= metric_helper.false_positive(outputs_val,labels_val)
            false_negatives+= metric_helper.false_negative(outputs_val,labels_val)
            
                        
#     print("Total val", total_val)
#     print("Correct val", correct_val)
#     print("Accuracy", correct_val * 100 / total_val)
    
    
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


# In[27]:


model = vgg16
if 1==1:
    total_val=0
    correct_val=0
    master_conf1=np.zeros((15,15))
    
    with torch.no_grad():
        for data_val in val_loader:
                images_val=data_val[0]
                labels_val = data_val[1]
                #labels_val = torch.tensor(labels, dtype=torch.long, device=computing_device)
                images_val, labels_val = images_val.to(computing_device), labels_val.to(computing_device)
                outputs_val = model(images_val)
                
                
                for label,prediction in zip(labels_val.cpu().numpy(),(outputs_val>=0.5).cpu().numpy()):
                    
                    conf=np.zeros((15,15))
                    i=0
                    for l,p in zip(label,prediction):
                        if(l+p==0):
                            conf[14][14]=1
                        if(l+p==2):
                            conf[:,i]=np.hstack((prediction,0))
                        if(p==1 and l==0):
                            
                            conf[:,14]=np.hstack((prediction,0))
                        if(p==0 and l==1):
                            conf[:,i]=np.hstack((prediction,0))
                            conf[14][i]=1
                        i+=1
                    master_conf1+=conf


# In[28]:


import pandas as pd

f={0: "Atelectasis", 1: "Cardiomegaly", 2: "Effusion", 
                3: "Infiltration", 4: "Mass", 5: "Nodule", 6: "Pneumonia", 
                7: "Pneumothorax", 8: "Consolidation", 9: "Edema", 
                10: "Emphysema", 11: "Fibrosis", 
                12: "Pleural_Thickening", 13: "Hernia"}
cols=f.values()
c=['Atelectasis', 'CardioM.', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pnem.', 'Pneumoth.', 'Cons.', 'Edema', 'Emphy.', 'Fibrosis', 'PleuralTh', 'Hernia',"NA"]
df2=pd.DataFrame(columns=['Prediction'],data=['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thick.', 'Hernia',"NA"])
# df2.ignoreIndex(True)
df=pd.DataFrame(columns=c,data=master_conf1)


# In[29]:


pd.concat([df2,df],axis=1)


# In[ ]:


N_minibatch_loss = 0.0 

for minibatch_count, (images, labels) in enumerate(val_loader, 0):

        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, labels = images.to(computing_device), labels.to(computing_device)
        labels1=torch.tensor(labels, dtype=torch.long, device=computing_device)

        # Perform the forward pass through the network and compute the loss
        outputs = model(images)
        v_o,i_o=torch.max(outputs,1)
        v_l, i_l=torch.max(labels1,1)
        loss = criterion(outputs,labels)
        


        # Add this iteration's loss to the total_loss
        total_loss.append(loss.item())
        N_minibatch_loss += loss
        
        N_minibatch_loss /= N
        print('Epoch %d, average minibatch %d loss: %.3f' %
                (epoch + 1, minibatch_count, N_minibatch_loss))
        


# In[ ]:





# In[ ]:


#Plot cross-entropy loss
X=np.linspace(0,config['epochs'],60).reshape((60,1))
Y=Loss_valid.reshape((60,1))
Z=Loss_training.reshape((60,1))
fig = plt.figure(figsize=(20,10))
ax = plt.subplot(111)
ax.plot(X,Y,label='Training')
ax.plot(X,Z,label='Validation')
plt.ylabel('Cross-entropy Loss')
plt.xlabel('Epochs')
plt.title('Training and Validation Loss')
ax.legend()
plt.show()
#plot accuracy
Y_v = A_valid.reshape((60,1))
Y_tr = A_training.reshape((60,1))
fig = plt.figure(figsize=(20,10))
ax = plt.subplot(111)
ax.plot(X,Y_tr,label='Training')
ax.plot(X,Y_v,label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title('Validation and Training Accuracy')
ax.legend()
plt.show()

