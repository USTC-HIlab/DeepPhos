import functools
import itertools
import os
import random
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from keras.layers import Dense, Activation, Flatten, Dropout, Reshape
from keras.layers import Conv1D,Conv2D, MaxPooling2D
from keras.models import Sequential,Model
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.optimizers import Adam,SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import copy

def predict_for_deepphos(train_file_name,sites,predictFrame = 'general',
                         hierarchy=None, kinase=None):
    '''

    :param train_file_name: input of your prdict file
                            it must be a .csv file and theinput format  is proteinName, postion,sites, shortseq
    :param sites: the sites predict: site = 'S','T' OR 'Y'
    :param predictFrame: 'general' or 'kinase'
    :param hierarchy: if predictFrame is kinse: you must input the hierarchy:
            group,family,subfamily,kinase to choose corresponding model
    :param kinase: kinase name
    :return:
     a file with the score
    '''


    win1 = 51
    win2 = 33
    win3 = 15
    from methods.dataprocess_predict import getMatrixInput
    [X_test1,y_test,ids,position] = getMatrixInput(train_file_name, sites, win1)
    [X_test2,_,_,_] = getMatrixInput(train_file_name, sites, win2)
    [X_test3,_,_,_]  = getMatrixInput(train_file_name, sites, win3)

    print X_test1.shape
    print len(position)

    from methods.model_n import model_net
    model = model_net(X_test1, X_test2, X_test3, y_test,nb_epoch = 1)

    #load model weight
    if predictFrame == 'general':
        outputfile = 'general_{:s}'.format(site)
        if site == ('S','T'):
            model_weight = './models/model_general_S,T.h5'
        if site == 'Y':
            model_weight = './models/model_general_Y.h5'


    if predictFrame == 'kinase':
        outputfile = 'kinase_{:s}_{:s}'.format(hierarchy, kinase)
        model_weight = './models/model_{:s}_{:s}.h5'.format(hierarchy, kinase)
    print model_weight
    model.load_weights(model_weight)
    predictions_t = model.predict([X_test1, X_test2, X_test3])
    results_ST = np.column_stack((ids, position,predictions_t[:, 1]))

    result = pd.DataFrame(results_ST)
    result.to_csv(outputfile + "prediction_phosphorylation.txt", index=False, header=None, sep='\t',
                  quoting=csv.QUOTE_NONNUMERIC)
if __name__ == '__main__':
    train_file_name = 'test data.csv'
    site = 'S','T'
    predict_for_deepphos(train_file_name, site, predictFrame='kinase',
                         hierarchy='group', kinase='AGC')




