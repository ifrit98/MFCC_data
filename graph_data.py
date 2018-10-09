#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 17:32:44 2018

@author: jason st. george
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

SAMPLES = 100

RED_COLOR    = "#E25151"
ORANGE_COLOR = "#F45328"
GREEN_COLOR  = "#A0CE94" 
BLUE_COLOR   = "#738992"
PURPLE_COLOR = "#B300C4"
REAL_GREEN   = "#007800"

GRAPH_NAME = "Mel Frequency Cepstrum Coefficients"

def get_data(fn):
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

def calc_mean(xs):
    means = []
    for x in xs:
      means.append(sum(x) / len(x))      
    return means
        
def calc_sigma(xs):
    sigmas = []
    for x in xs:
        sigmas.append( np.sqrt( np.var( x ) ) )

    return sigmas         

def covariance(xs, file):
    print('Covariance matrices from file: ' + file)
    print('Covariance matrix of Hyla Minuta:')
    print(np.cov([xs[0], xs[1]]))
    print('Covariance matrix of Hypsiboas Cinerascens:')
    print(np.cov([xs[2], xs[3]]))
    print()

def stats(data, fn):
    means = calc_mean(data)
    stds = calc_sigma(data)
    
    for i in range(4):
        means[i] = means[i] * 1000
        stds[i] = stds[i] * 1000
        print('Mean ' + str(i) + ': ' + str(means[i]))
        print('Standard Dev ' + str(i) + ':  ' + str(stds[i]))


def histograms(data, fn):
    colors = [ORANGE_COLOR, GREEN_COLOR, BLUE_COLOR, PURPLE_COLOR]
    labels = ["Hyla Minuta", "Hyla Minuta", "Hypsiboas Cinerascens", "Hypsiboas Cinerascens"]
    
    for i in range(4):
        x = data[i]
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(x, bins=30, density=1, facecolor=colors[i], alpha=1)
        ax.set_xlabel("MFCC_17") if i > 1 else ax.set_xlabel("MFCC_10")
        ax.set_ylabel("Frequency")
        ax.set_title(str(labels[i]) + ': ' + str(fn))
        
        ax.grid(True)
        fig.show()
        fig.savefig('./histogram' + str(i) + fn[:-3] )
    
def line_graphs(data, fn):
    for d in data:
        d.sort()
        print()
    
#    labels = ["Hyla Minuta", "Hyla Minuta", "Hypsiboas Cinerascens", "Hypsiboas Cinerascens"]

    x1 = np.arange(len(data[0])) 
    x2 = np.arange(len(data[2]))
    
    fig = plt.figure(figsize=(11,8))
    ax1 = fig.add_subplot(111)
    
    ax1.plot(x1, data[0], label='Hyla Minuta feat 1')
    ax1.plot(x1, data[1], label='Hyla Minuta feat 2')
    ax1.plot(x2, data[2], label='Hypsiboas Cinerascens feat 1')
    ax1.plot(x2, data[3], label='Hypsiboas Cinerascens feat 2')
    ax1.legend(loc=2)
    ax1.set_xlabel('Examples')
    ax1.set_ylabel('Frequency Coefficient (MFCC)')
    ax1.set_title(fn)
    
    colors = [BLUE_COLOR, PURPLE_COLOR, ORANGE_COLOR, REAL_GREEN]
    for i,j in enumerate(ax1.lines):
        j.set_color(colors[i])
    fig.savefig('./line_graph' + fn[:-3])
    # As individual graphs    
#    for i in range(4):
#        fig = plt.figure(figsize=(11,8))
#        ax1.plot(x1, data[i], label=labels[i])        
    
def box_plots(data, fn):
    plt.figure()
    plt.boxplot(data)
    plt.xticks([1, 2, 3, 4], ['HM feat 1', 'HM feat 2', 'HC feat 1', 'HC feat 2'])    
    plt.ylabel('Frequenct Coefficients (MFCCs)')
    plt.title(fn)
    plt.savefig('./box_plot' + fn[:-3])
    plt.show()    

def bar_graph(data, fn):        
    means = calc_mean(data)
    stds = calc_sigma(data)
    
    for i in range(4):
        means[i] = means[i] * 1000
        stds[i] = stds[i] * 1000

    ind = np.arange(4)    
    width = 0.35      
    
    p1 = plt.bar(ind, means, width, yerr=stds, color=['red','blue','red','blue'])
    
    plt.ylabel('Frequency Coefficient\n Scaled (10^-3)')
    plt.xlabel('Features')
    plt.title('Mel Frequency Cepstrum Coefficients: ' + str(fn))
    plt.xticks(ind, ('HM MFCC_10', 'HM MFCC_10', 'HC MFCC_17', 'HC MFCC_17'))
    plt.yticks(np.arange(-150, 150, 50))
    plt.legend((p1[0], p1[3]), ('HM', 'HC'))
    plt.savefig('./bar_graph_' + fn[:-3])
    plt.show()

    
def scatter(x1, y1, x2, y2, fn):
    fig, ax = plt.subplots()
    
    ax.scatter(x1, y1, marker="o", color=ORANGE_COLOR, label = "Hyla Minuta")
    ax.scatter(x2, y2, marker="x", color=PURPLE_COLOR, label = "Hypsiboas Cinerascens")    

    ax.set_title("Mel Frequency Cepstrum Coefficients: " + fn)

    ax.legend(loc='upper left')
    
    ax.set_xlabel("MFCCs_10")
    ax.set_ylabel("MFCCs_17")
    
    ax.set_xlim([-0.75, 0.75])
    ax.set_ylim([-0.75, 0.75])
    
    fig.show()
    fig.savefig("./" + GRAPH_NAME + fn[:-3])
    
def main():
    files = ['Frogs-subsample.csv', 'Frogs.csv']
    for f in files:
        x1, y1, x2, y2 = get_data(f)
        covariance([x1, y1, x2, y2], f)
        stats([x1, y1, x2, y2], f)
        box_plots([x1,y1,x2,y2], f)
        bar_graph([x1, y1, x2, y2], f)
        scatter(x1, y1, x2, y2, f)
        histograms([x1, y1, x2, y2], f)
        line_graphs([x1, y1, x2, y2], f)
        print()
    
if __name__ == '__main__':
    main()
    