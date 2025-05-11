import torch
import time
# import necessary libraries
import torch.nn as nn
import torch.optim as optim
import numpy as np
#cronometre start
start_time = time.time()

#input training data
input_data = np
#generate traning data
x = torch.arange(1,10,0.1).reshape(-1,1).float()
y = 2*x
# define a simple neuron model
class Neuron(torch.nn.Module):
    def __init__(self):
        super(Neuron, self).__init__()
        # a single linear layer (y = wx + b)
        self.linear = nn.Linear(1, 1)  # Input size = 1, Output size = 1

    def forward(self, x):
        # forward pass through the linear layer
        return self.linear(x)

# create an instance of the neuron
neuron = Neuron()

# define a loss function (Mean Squared Error)
criterion = nn.MSELoss()

# define an optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(neuron.parameters(), lr=0.01)

# training data (x: input, y: target output)
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=False)
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]], requires_grad=False)

# training loop
for epoch in range(100):  # Number of epochs
    # Forward pass
    outputs = neuron(x_train)
    loss = criterion(outputs, y_train)

    # backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

        #show the learning rate
        print("\nfinal parameters:")
        print(f"weights:{neuron.linear.weight.item():.4f}")
        print(f"bias:{neuron.linear.bias.item():.4f}")

# test the trained neuron
for test_input in np.arange(1.0, 4.0, 1.0):
    test_tensor = torch.tensor([[test_input]], dtype=torch.float32)
    predicted_output = neuron(test_tensor)
    print(f'Input: {test_input}, Predicted Output: {predicted_output.item()}') 


#tensor in axes
for i in range(0,4):
    x_train[i]=x_train[i]*2
    y_train[i]=y_train[i]*2
# print the modified tensors        
print("Modified x_train:", x_train)
print("Modified y_train:", y_train)


#cronometre end and print elapsed time
elpased_time= time.time()-start_time
print(f"Elapsed time:{elpased_time:.2f}seconds")

# Documentation:
# This script demonstrates a simple example of a single neuron implemented in PyTorch.
# The neuron is trained to learn a linear relationship (y = 2x) using a small dataset.
# The model consists of a single linear layer, and the training process uses MSE loss
# and SGD optimizer. After training, the neuron can predict outputs for new inputs.