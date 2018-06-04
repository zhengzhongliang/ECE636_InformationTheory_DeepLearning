import numpy as np
import matplotlib.pyplot as plt
#This script plots the accuracy of classifiers based on the result of 4_Classification_CENT.py


Acc = np.load('acc_array.npy')

plt.figure()
lines={}
colors = ['blue', 'royalblue', 'cornflowerblue', 'deepskyblue', 'turquoise', 'limegreen', 'greenyellow', 'springgreen',
          'yellow', 'tomato']

labels = ['Logistic\nRegression','Naive\nBayes','Linear\nSVM','SVM','Decision\nTree','Random\nForest']
for i in np.arange(len(Acc)):
    plt.subplot(2,3,i+1)
    x_range = np.arange(9)+2
    lines[i], = plt.plot(x_range, Acc[i,:], color = colors[i], label = labels[i])
    plt.xlabel('number of classes')
    plt.ylabel('classification accuracy')
    plt.xlim((2,9))
    plt.ylim((0,1))
    plt.legend(handles=[lines[i]], fontsize=9, loc='lower left')
    plt.xticks(x_range, x_range)

plt.tight_layout()
plt.show()
