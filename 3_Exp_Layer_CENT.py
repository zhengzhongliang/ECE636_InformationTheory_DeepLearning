# this script is used to calculate the CENT for all filters in the layer.
# Besure to run 2_Exp_Filter_Discriminator.py before you run this script.

import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os
import pickle as pkl

generate_data = False

def Generate_Layer_CENT(Filter_CENT_Dict):
    layer_CENT_dict = {}

    for i in np.arange(10):   #loop over all classes
        filter_CENT_array = Filter_CENT_Dict[i]
        layer_CENT_array = np.zeros((len(filter_CENT_array),2))
        for j in np.arange(len(filter_CENT_array)):   #loop over all samples in class i
            layer_CENT_array[j,0] = np.sum(filter_CENT_array[j,:,0])/10.0
            layer_CENT_array[j,1] = np.sum(filter_CENT_array[j,:,1])/10.0

        layer_CENT_dict[i] = layer_CENT_array

    return layer_CENT_dict

save_path = 'layer_output_dict'
if generate_data:
    f1 = open("filter_output_dict/CENT_train.pkl", 'rb')
    CENT_train = pkl.load(f1)
    f1.close()

    f2 = open("filter_output_dict/CENT_test.pkl", 'rb')
    CENT_test = pkl.load(f2)
    f2.close()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    layer_CENT_train = Generate_Layer_CENT(CENT_train)
    layer_CENT_test = Generate_Layer_CENT(CENT_test)

    f1 = open(save_path + "/layer_CENT_train.pkl", "wb")
    pkl.dump(layer_CENT_train, f1)
    f1.close()

    f2 = open(save_path + "/layer_CENT_test.pkl", "wb")
    pkl.dump(layer_CENT_test, f2)
    f2.close()

    print('layer dict stored to ' + save_path)

else:
    f1 = open(save_path + "/layer_CENT_train.pkl", 'rb')
    layer_CENT_train = pkl.load(f1)
    f1.close()

    f2 = open(save_path + "/layer_CENT_test.pkl", 'rb')
    layer_CENT_test = pkl.load(f2)
    f2.close()

    colors = ['blue', 'royalblue', 'cornflowerblue', 'deepskyblue', 'turquoise', 'limegreen', 'greenyellow',
              'springgreen', 'yellow', 'tomato']

    plt.figure(figsize=(5,5))
    dots = {}
    plt.title("Training Data")
    for i in np.arange(10):  #loop over all classes
        dots[i] = plt.scatter(layer_CENT_train[i][:,0], layer_CENT_train[i][:,1], alpha=0.2, color = colors[i], label="digit "+str(i))
    plt.legend(handles = [dots[i] for i in np.arange(10)], loc="lower right")
    plt.xlabel("CENT Layer 1")
    plt.ylabel("CENT Layer 2")
    plt.xlim((1,8))
    plt.ylim((1,8))

    plt.figure(figsize=(20,8))
    dots = {}
    plt.title("Training Data")
    for i in np.arange(10):  # loop over all classes
        plt.subplot(2,5,i+1)
        dots[i] = plt.scatter(layer_CENT_train[i][:, 0], layer_CENT_train[i][:, 1], alpha=0.2, color=colors[i],
                              label="training set, digit " + str(i))
        plt.xlabel("CENT Layer 1")
        plt.ylabel("CENT Layer 2")
        plt.xlim((1, 8))
        plt.ylim((1, 8))
        plt.legend(handles=[dots[i]], loc="lower right")
    plt.tight_layout()

    plt.figure(figsize=(5, 5))
    dots = {}
    plt.title("Testing Data")
    for i in np.arange(10):  # loop over all classes
        dots[i] = plt.scatter(layer_CENT_test[i][:, 0], layer_CENT_test[i][:, 1], alpha=0.2, color=colors[i],
                              label="digit " + str(i))
    plt.legend(handles=[dots[i] for i in np.arange(10)], loc="lower right")
    plt.xlabel("CENT Layer 1")
    plt.ylabel("CENT Layer 2")
    plt.xlim((1, 8))
    plt.ylim((1, 8))

    plt.figure(figsize=(20, 8))
    dots = {}
    plt.title("Testing Data")
    for i in np.arange(10):  # loop over all classes
        plt.subplot(2,5,i+1)
        dots[i] = plt.scatter(layer_CENT_test[i][:, 0], layer_CENT_test[i][:, 1], alpha=0.2, color=colors[i],
                              label="testing set, digit " + str(i))
        plt.xlabel("CENT Layer 1")
        plt.ylabel("CENT Layer 2")
        plt.xlim((1, 8))
        plt.ylim((1, 8))
        plt.legend(handles=[dots[i]], loc="lower right")
    plt.tight_layout()

    plt.show()