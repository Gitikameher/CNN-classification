from baseline_cnn import *
from baseline_cnn import BasicCNN


# Setup: initialize the hyperparameters/variables
num_epochs = 5          # Number of full passes through the dataset
batch_size = 16          # Number of samples in each minibatch
learning_rate = 0.001
seed = np.random.seed(1) # Seed the random number generator for reproducibility
p_val = 0.1              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing
momentum=0.9

#TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
transform = transforms.Compose([transforms.Resize(512),transforms.ToTensor()])

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

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
                                                             shuffle=True, show_sample=False,
                                                             extras=extras)

# Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
model = BasicCNN()
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

#TODO: Define the loss criterion and instantiate the gradient descent optimizer
criterion = model.loss_function()
#TODO - loss criteria are defined in the torch.nn package

#TODO: Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.SGD(model.parameters(),lr = learning_rate, momentum = momentum)
#TODO - optimizers are defined in the torch.optim package

# Track the loss across training
total_loss = []
avg_minibatch_loss = []

# Begin training procedure
for epoch in range(num_epochs):

    N = 50
    N_minibatch_loss = 0.0

    # Get the next minibatch of images, labels for training
    for minibatch_count, (images, labels) in enumerate(train_loader, 0):
        
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, labels = images.to(computing_device), labels.to(computing_device)

        # Zero out the stored gradient (buffer) from the previous iteration
        optimizer.zero_grad()

        # Perform the forward pass through the network and compute the loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Automagically compute the gradients and backpropagate the loss through the network
        loss.backward()

        # Update the weights
        optimizer.step()    
        # Add this iteration's loss to the total_loss
        total_loss.append(loss.item())
        N_minibatch_loss += loss
        
        # TODO: Implement validation #with torch.no_grad():
        correct_val = 0
        total_val= 0
        
        # correct_val,total_val = correct_val.to(computing_device),total_val.to(computing_device)
        with torch.no_grad():
            for data_val in test_loader:
                images_val, labels_val = data_val
                labels_val = labels_val.type(torch.uint8)
                images_val,labels_val = images_val.to(computing_device),labels_val.to(computing_device)
                outputs_val = model(images_val)
                # _, predicted_val = torch.greater(outputs_val,1)
                predicted_val = torch.ge(outputs_val,0.5)  #>=0.5
                # print(predicted_val)
                # print(predicted_val[0])
                #print("Predicted_val",  predicted_val.size())
                #print("Labels_val",labels_val.size())
                # print(predicted_val.shape)
                # print(labels_val)
                # print(labels_val[0])
                # total_val += labels_val.size(0)
                #change this later, temporary
                total_val += labels_val.numel()
                # print("total_val",total_val)
                correct_val += (predicted_val == labels_val).sum().item()
                # print("correct_val",correct_val)
        # print("Total val",total_val)
        # print("Correct val",correct_val)
        print(f"Accuracy @ Epoch{epoch} : {correct_val*100/total_val}")        
        ##Val End
        
        print("Val Complete")
        
        if minibatch_count % N == 49:
            #Print the loss averaged over the last N mini-batches
            N_minibatch_loss /= N
            print('Epoch %d, average minibatch %d loss: %.3f' % (epoch + 1, minibatch_count+1, N_minibatch_loss))
            # Add the averaged loss over N minibatches and reset the counter
            avg_minibatch_loss.append(N_minibatch_loss)
            N_minibatch_loss = 0.0

    print("Finished", epoch + 1, "epochs of training")
print("Training complete after", epoch, "epochs")