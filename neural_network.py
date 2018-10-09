#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 21:54:19 2018

@author: jason
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import matplotlib.ticker as ticker
import sys

ORANGE_COLOR = "#F45328"
BLUE_COLOR   = "#738992"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))

data = np.load('Hastie-data.npy')
x = data[:,0]

EPOCHS = 1024

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def get_data(fn):
    data = np.load('Hastie-data.npy')   
    
    x1 = data[:,0]
    x2 = data[:,1]
    y = data[:,2]
    
    take_test = int(len((y/2)) * 0.1)
           
    x1_train = x1[take_test:(len(y)-take_test)]
    x1_test = x1[:take_test]
    x1_test = np.append(x1_test, x1[len(y)-take_test:])
    x2_train = x2[take_test:(len(y)-take_test)]
    x2_test = x2[:take_test]
    x2_test = np.append(x2_test, x2[len(y)-take_test:])
    y_train = y[take_test:(len(y)-take_test)]
    y_test = y[:take_test]
    y_test = np.append(y_test, y[len(y)-take_test:])
            
     
    train = [ (x1_train[i], x2_train[i]) for i in range(len(x1_train)) ]
    train_labels = y_train
    test = [ (x1_test[i], x2_test[i]) for i in range(len(x1_test)) ]
    test_labels = y_test
    
    train_targ = torch.LongTensor(train_labels)
    train_input = torch.FloatTensor(train)
    
    test_targ = torch.LongTensor(test_labels)
    test_input = torch.FloatTensor(test)
    
    train = torch.utils.data.TensorDataset(train_input, train_targ)
    test = torch.utils.data.TensorDataset(test_input, test_targ)
    
    return train, test


def get_data_split(fn):
    data = np.load('Hastie-data.npy')   

    class_0 = data[data[:,2] == 0]
    class_1 = data[data[:,2] == 1]
    
    return class_0[:,0], class_0[:,1], class_1[:,0], class_1[:,1]

def train_model(net, trainloader, optimizer, criterion):      
    saved_losses = []
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
    
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad()
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
        
        saved_losses.append(running_loss / 781)
        print('Epoch ' + str(epoch + 1) + ', loss: ' + str(
            running_loss / 781))
        running_loss = 0.0
    
    print('Finished Training')

    weights = list(net.parameters())
    
    with open('./myweightsQ3.txt', 'w') as f:
        for item in weights:
            f.write("%s\n" % item)         
        
    torch.save(net.state_dict(), './q3model_Hastie')
    
    return saved_losses, weights
            

def plot(saved_losses):
    fig, ax = plt.subplots()

    x = np.linspace(1, EPOCHS, EPOCHS)
    saved_losses = np.array(saved_losses)

    ax.set_title("Average Model Loss over Epochs")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average Loss")

    tick_spacing = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.plot(x, saved_losses, color='purple', marker=".")
    fig.savefig('./model_loss_hastie')


def plot_nonlinear_DB(net, testloader, trainloader):    
    dataiter = iter(trainloader)
    inputs, labels = dataiter.next()
    
    colors = [ORANGE_COLOR, BLUE_COLOR]

    X = np.load('Hastie-data.npy')
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max),
                         np.arange(y_min, y_max))    
    
    Z = net(inputs) 
    inputs, labels = inputs.to(device), labels.to(device)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')
    
    plt.scatter(X[:, 0], X[:, 1], c=colors, cmap=plt.cm.Paired)

    
def plot_DB(net, data, fn):
    w = list(net.parameters())
    fig, ax = plt.subplots()
    
    ax.scatter(data[0], data[1], marker="o", color="#F45328", label = "Class 0")
    ax.scatter(data[2], data[3], marker="x", color="#B300C4", label = "Class 1")    

    ax.set_title("Hastie-data.npy Decision Boundary: ")

    ax.legend(loc='upper left')
    
    ax.set_xlabel("MFCCs_10")
    ax.set_ylabel("MFCCs_17")
    
    ax.set_xlim([-4, 5])
    ax.set_ylim([-4, 5])
    
    fig.show()
    
    print(w)
    
    w0 = w[0].data.numpy()
    w1 = w[1].data.numpy()
    
    print("Final gradient descent:", w)

    x_axis = np.linspace(-6, 6, 100)
    y_axis = -(w1[0] + x_axis*w0[0][0]) / w0[0][1]
    line_up, = plt.plot(x_axis, y_axis,'r--', label='Regression line')
    plt.legend(handles=[line_up])
    plt.xlabel('X(1)')
    plt.ylabel('X(2)')
    plt.show()        
    
    fig.savefig('./decision_boundHastieA')
    plt.savefig('./decision_boundHastieB')
    
    
def accuracy(net, testloader, classes, test_len):
    net.load_state_dict(torch.load('q3model_Hastie'))
    
    dataiter = iter(testloader)
    inputs, labels = dataiter.next()
    
    print('\nGround Truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)
    
    _, predicted = torch.max(outputs, 1)
    
    print('Predicted:    ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('\nAccuracy of the network on the %d test inputs: %f %%\n' % (test_len, (100 * correct / total)))
    
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(2):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        
        

def main():
    assert len(sys.argv) > 1, "Usage: \n q3.py train\n q3.py test"
    if sys.argv[1] == 'test':
        flag = False
    else: 
        flag = True
        
    fn = "Hastie-data.npy"
    print("Getting data for " + fn + "...")
    train, test = get_data(fn)
           
    print("Building train/test loader for " + fn + "...")
    trainloader = torch.utils.data.DataLoader(train, batch_size=10,
                                              shuffle=True, num_workers=2)
    
    testloader = torch.utils.data.DataLoader(test, batch_size=10,
                                                 shuffle=True, num_workers=2)
    
    classes = ('A', 'B')
        
    print("Initializing net for " + fn + "...")
    net = NeuralNet(2, 8, 2).to(device)
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
    
    if flag:
        print("Training for " + fn  + "...")
        saved_loss, weights = train_model(net, trainloader, optimizer, criterion)
    
        print("Plotting for " + fn  + "...")
        plot(saved_loss)
    
    print("Computing accuracy for " + fn  + "...")
    accuracy(net, testloader, classes, len(test))
                
#    plot_DB(net, get_data_split(fn), fn)
    plot_nonlinear_DB(net, testloader, trainloader)
        
if __name__ == '__main__':
    main()