import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline 
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F

#Generates a set of 100 eigenvalues randomly from -6 to 6
eigen_arr = []
for i in range(1,101):
    eigen_arr.append(random.uniform(-6,6))

 

#single hidden layer neural network
class GaussNN(nn.Module):
        
        def __init__(self, n_feature, n_hidden, n_output):
            super(GaussNN, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden) 
            self.predict = torch.nn.Linear(n_hidden, n_output)   

        def forward(self, x):
            x = F.elu(self.hidden(x))    
            x = self.predict(x)             
            return x
        
        
G = GaussNN(n_feature=1, n_hidden=10, n_output=1) 
#lr = 0.75-0.95 usually provides nicest results` 
optimizer = torch.optim.SGD(G.parameters(), lr=0.075)
loss_func = torch.nn.MSELoss()   
 
#Standard deviation of 0.1 ev
sigma = 0.1  

Denom = (sigma)**2


#Defining Gaussian data inside 5 standard deviations, or 
# 99.9999426% of all data. It could be changed to 3 or 4,
# Serves to exclude all data in the gaussian where y = 0 
# inside the (-6,6) range while retaining as much nonzero data 
# as possible;
x = torch.unsqueeze(torch.linspace(-5*sigma, 5*sigma, 500), dim=1)  
#In this case it's defined over (-0.5,0.5) of the mean of each gaussian dst

x = Variable(x)

EPOCHS = 3000
y_pred = [] 
arr_pred = []
 
#The model is trained over every eigenvalue in eigen_arr    
for eigenvalues in eigen_arr: 
    #Pytorch gaussian distribution defined 
    y_input = torch.exp(-x.pow(2)/(2*Denom), out=None)
    
    for i in range(1,EPOCHS+1): 
            prediction = G(x)       
            loss = loss_func(prediction, y_input)      
            optimizer.zero_grad()    
            loss.backward()          
            optimizer.step()          
    y_pred.append(prediction.data)
arr_pred.append(y_pred)


#Values of x corresponding to y_pred 
X = []
for eigenvalues in eigen_arr:
    X.append((x + eigenvalues).tolist())

#Set of predictions for all eigenvalues  
Y = []
for i in range(len(eigen_arr)):
    Y.append(arr_pred[0][i].tolist())

results = pd.DataFrame({'x': X,
                        'y': Y
})


#Save the csv file to your folder.
#There will be two columns, x(eigenvalues) and y(preds)
results.to_csv('FILE PATH/results.csv')
