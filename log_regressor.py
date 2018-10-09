#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:49:26 2018

@author: jason
"""
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data
import matplotlib.ticker as ticker
import csv
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))

EPOCHS = 512

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(2,2)

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        
        return x


def get_data(fn):
    x1, x2, y = [], [], []
    with open(fn,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')        
        next(plots, None)
        
        for row in plots:
            x1.append(float(row[0]))
            x2.append(float(row[1]))
            y.append(1) if row[2] == "HypsiboasCinerascens" else y.append(0)
    
    csvfile.close()
    
    z = 50 if fn == 'Frogs.csv' else 4

    
    train = [ (x1[i], x2[i]) for i in range(z, len(y)-z) ]
    train_labels = [ y[i] for i in range(z, len(y)-z) ]
    
    train.extend( [ (x1[i], x2[i]) for i in range(z, len(y)-z) ] )
    train_labels.extend( [ y[i] for i in range(z, len(y)-z) ] )     

    test = [ (x1[i], x2[i]) for i in range(z) ]
    test_labels = [ y[i] for i in range(z) ]
    
    test.extend( [ (x1[i], x2[i]) for i in range(len(y)-1, len(y)-z, -1) ] )
    test_labels.extend( [ y[i] for i in range(len(y)-1, len(y)-z, -1) ] )

    train_targ = torch.LongTensor(train_labels)
    train_input = torch.FloatTensor(train)
    
    test_targ = torch.LongTensor(test_labels)
    test_input = torch.FloatTensor(test)
    
    train = torch.utils.data.TensorDataset(train_input, train_targ)
    test = torch.utils.data.TensorDataset(test_input, test_targ)
    
    return train, test
    

def get_data_split(fn):
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    
    with open(fn,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')        
        next(plots, None)  # skip the headers
        for row in plots:
            if row[2] == "HypsiboasCinerascens":
                x2.append(float(row[0]))
                y2.append(float(row[1]))
            else:
                x1.append(float(row[0]))
                y1.append(float(row[1]))
        
    return x1, y1, x2, y2
        

def train_model(net, trainloader, optimizer, criterion, fn):      
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
    
    with open('./myweights_' + fn[:-3] + '.txt', 'w') as f:
        for item in weights:
            f.write("%s\n" % item)       
            
    torch.save(net.state_dict(), './q2model_' + fn)
    
    return saved_losses, weights
            

def plot(saved_losses):
    fig, ax = plt.subplots()

    x = np.linspace(1, EPOCHS, EPOCHS)
    saved_losses = np.array(saved_losses)

    ax.set_title("Average Model Loss over Epochs")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average Loss")

    # Adjust x-axis ticks
    tick_spacing = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.plot(x, saved_losses, color='purple', marker=".")
    fig.savefig('./model_loss')

    
def plot_DB(net, data, fn):
    w = list(net.parameters())
    fig, ax = plt.subplots()
    
    ax.scatter(data[0], data[1], marker="o", color="#F45328", label = "Hyla Minuta")
    ax.scatter(data[2], data[3], marker="x", color="#B300C4", label = "Hypsiboas Cinerascens")    

    ax.set_title("MFCC Decision Boundary: " + str(fn))

    ax.legend(loc='upper left')
    
    ax.set_xlabel("MFCCs_10")
    ax.set_ylabel("MFCCs_17")
    
    ax.set_xlim([-0.75, 0.75])
    ax.set_ylim([-0.75, 0.75])
    
    fig.show()
    
    w0 = w[0].cpu().data.numpy()
    w1 = w[1].cpu().data.numpy()
    
    print("Final gradient descent:", w)

    x_axis = np.linspace(-6, 6, 100)
    y_axis = -(w1[0] + x_axis*w0[0][0]) / w0[0][1]
    line_up, = plt.plot(x_axis, y_axis,'r--', label='Regression line')
    plt.legend(handles=[line_up])
    plt.xlabel('X(1)')
    plt.ylabel('X(2)')
    
    fig.savefig('./decision_boundA_' + fn[:-3])
    plt.savefig('./decision_boundB_' + fn[:-3])
    
    plt.show()        
    
    
def accuracy(net, testloader, classes, test_len, fn):
    net.load_state_dict(torch.load('q2model_' + fn))
    
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
    files = ['Frogs.csv']
    assert len(sys.argv) > 1, "Usage: \n q2.py train\n q2.py test"
    if sys.argv[1] == 'train':
        flag = True
    else: 
        flag = False
    
    for fn in files:
        print("Getting data for " + fn + "...")
        train, test = get_data(fn)
               
        print("Building train/test loader for " + fn + "...")
        trainloader = torch.utils.data.DataLoader(train, batch_size=12,
                                                  shuffle=True, num_workers=2)
        
        testloader = torch.utils.data.DataLoader(test, batch_size=12,
                                                     shuffle=True, num_workers=2)
        
        classes = ('0', '1')
            
        print("Initializing net for " + fn + "...")
        net = Net()
        net.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)
        
        if flag:
            print("Training for " + fn  + "...")
            saved_loss, weights = train_model(net, trainloader, optimizer, criterion, fn)
        
            print("Plotting for " + fn  + "...")
            plot(saved_loss)
            
            print("Computing accuracy for " + fn  + "...")
            
            accuracy(net, testloader, classes, len(test), fn)
 
            plot_DB(weights, get_data_split(fn), fn)
            
        else:
            accuracy(net, testloader, classes, len(test), fn)
            plot_DB(net, get_data_split(fn), fn)            
        
                    
if __name__ == '__main__':
    main()
    