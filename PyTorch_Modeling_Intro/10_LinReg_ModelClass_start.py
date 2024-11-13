#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

#%% model class
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


#defining parameters and making an instance of the model  
input_dim =1
output_dim = 1
model = LinearRegressionTorch(input_dim, output_dim)


    
# %%loss function
loss_function = nn.MSELoss()
# %%optimizer
LR = 0.025
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
#test differenet value from too large 0.1 to too small 0.001

# %%perform training
losses, slope, bias = [], [], []

num_epochs = 2000

for epoch in range(num_epochs):
    #set the gradients to zero
    optimizer.zero_grad()

    #forward pass
    y_pred = model(X)

    #calculate loss
    loss = loss_function(y_pred, y_true)

    #compute gradients
    loss.backward()

    #update weights
    optimizer.step()

    #get the parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])

    #store the loss
    losses.append(float(loss.data))

    #print the loss
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, loss is {loss.data}')


# %%visualizing model training, realtion between epochs and loss
sns.scatterplot(x= range(num_epochs), y=losses)


# %%visualizing model training, realtion between epochs and slope
sns.scatterplot(x= range(num_epochs), y=slope)

# %%visualizing model training, realtion between epochs and bias
sns.scatterplot(x= range(num_epochs), y=bias)

# %%
y_pred = model(X).data.numpy().reshape(-1)
sns.scatterplot(x=X_list, y=y_list)#true values
sns.lineplot(x=X_list, y=y_pred, color = 'red')#predicted values
# %%
