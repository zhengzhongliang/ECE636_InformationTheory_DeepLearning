# this script is used to generate the CENT for each filter in each layer. Besure that you have a trained CNN before you run this script.

import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os
import pickle as pkl

Generate_data = False

load_path_base = 'filter_output'
y_train = np.load(load_path_base + '/y_train.npy')
y_test = np.load(load_path_base + '/y_test.npy')
train_conv_1 = np.load(load_path_base + '/train_conv_1.npy')
train_conv_2 = np.load(load_path_base + '/train_conv_2.npy')
test_conv_1 = np.load(load_path_base + '/test_conv_1.npy')
test_conv_2 = np.load(load_path_base + '/test_conv_2.npy')

def Generate_Filter_Entropy(conv_1, conv_2, labels):
    CENT_dict = {}
    for i in np.arange(10): # loop over all labels
        conv_1_current = conv_1[labels==i]
        conv_2_current = conv_2[labels==i]

        coord_l1_l2 = np.zeros((len(conv_1_current), 10, 2))   # all samples with label i, 10 filters, 2 layers
        for j in np.arange(len(conv_1_current)):    # loop over all samples with label i
            for k in np.arange(10):    # loop over all filters
                rand_out_1 = conv_1_current[j, :, :, k]
                amin = np.amin(rand_out_1)
                amax = np.amax(rand_out_1)
                hist, edges = np.histogram(rand_out_1, bins=np.arange(amin, amax, (amax-amin)/256.0))
                probs = hist / 1.0 / 24 / 24
                CENT_1 = entropy(probs, base=2)
                coord_l1_l2[j, k, 0] = CENT_1

                rand_out_2 = conv_2_current[j, :, :, k]
                amin = np.amin(rand_out_2)
                amax = np.amax(rand_out_2)
                hist, edges = np.histogram(rand_out_2, bins=np.arange(amin, amax, (amax - amin) / 256.0))
                probs = hist / 1.0 / 24 / 24
                CENT_2 = entropy(probs, base=2)
                coord_l1_l2[j, k, 1] = CENT_2
        CENT_dict[i]= coord_l1_l2
        print('CENT Digit ',i,' Finished!')
    return CENT_dict

save_path = 'filter_output_dict'
if Generate_data:

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    CENT_train = Generate_Filter_Entropy(conv_1= train_conv_1, conv_2 =train_conv_2 , labels=y_train)
    CENT_test = Generate_Filter_Entropy(conv_1= test_conv_1, conv_2 =test_conv_2 , labels=y_test)

    f1 = open(save_path+"/CENT_train.pkl", "wb")
    pkl.dump(CENT_train, f1)
    f1.close()

    f2 = open(save_path+"/CENT_test.pkl", "wb")
    pkl.dump(CENT_test, f2)
    f2.close()

    print('dict stored to '+save_path)

else:
    f1 = open(save_path+"/CENT_train.pkl", 'rb')
    CENT_train = pkl.load(f1)
    f1.close()

    f2 = open(save_path+"/CENT_test.pkl", 'rb')
    CENT_test = pkl.load(f2)
    f2.close()
    # plot the filter discrimination, train
    for i in np.arange(10):   # loop over all filters
        plt.figure(figsize=(20,8))
        plt.title('CENT of Training Samples, Filter '+str(i))
        colors = ['blue', 'royalblue', 'cornflowerblue', 'deepskyblue', 'turquoise', 'limegreen', 'greenyellow', 'springgreen','yellow','tomato']
        for j in np.arange(10): # loop over all classes
            plt.subplot(2,5, j+1)
            dots = plt.scatter(CENT_train[j][:,i,0], CENT_train[j][:,i,1], alpha=0.2, color=colors[j], label='training set, filter '+str(i+1)+',  digit '+str(j))
            plt.legend(handles=[dots])
            plt.xlabel('CENT Layer 1')
            plt.ylabel('CENT Layer 2')
            plt.xlim((1,7))
            plt.ylim((1,7))
        plt.tight_layout()

    # plot the filter discrimination, test
    for i in np.arange(10):   # loop over all filters
        plt.figure(figsize=(20,8))
        plt.title('CENT of Testing Samples, Filter '+str(i))
        colors = ['blue', 'royalblue', 'cornflowerblue', 'deepskyblue', 'turquoise', 'limegreen', 'greenyellow', 'springgreen','yellow','tomato']
        for j in np.arange(10): # loop over all classes
            plt.subplot(2,5, j+1)
            dots = plt.scatter(CENT_test[j][:,i,0], CENT_test[j][:,i,1], alpha=0.2, color=colors[j], label='testing set, filter '+str(i+1)+',  digit '+str(j))
            plt.legend(handles=[dots])
            plt.xlabel('CENT Layer 1')
            plt.ylabel('CENt Layer 2')
            plt.xlim((1,7))
            plt.ylim((1,7))
        plt.tight_layout()

    plt.show()