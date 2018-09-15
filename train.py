import functools
import itertools
import os
import random

import matplotlib
matplotlib.use('Agg')
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

def train_for_deepphos(train_file_name,site,predictFrame = 'general',background_weight = None):
    '''

    :param train_file_name:  input of your train file
                                it must be a .csv file and theinput format  is label,proteinName, postion,sites, shortsequence,
    :param site: the sites predict: site = 'S','T' OR 'Y'
    :param predictFrame: 'general' or 'kinase'
    :param background_weight: the model you want load to pretrain new model
                                usually used in kinase training
    :return:
    '''


    win1 = 51
    win2 = 33
    win3 = 15
    from methods.dataprocess_train import getMatrixLabel
    X_train1 , y_train = getMatrixLabel(train_file_name, sites, win1)
    X_train2, _ = getMatrixLabel(train_file_name, sites, win2)
    X_train3, _ = getMatrixLabel(train_file_name, sites, win3)


    modelname = "general_{:s}".format(site)
    if predictFrame == 'general':
        modelname ="general_model_{:s}".format(site)


    if predictFrame == 'kinase':
        modelname = "kinase_model_{:s}".format(site)


    from methods.model_n import model_net

    model = model_net(X_train1, X_train2, X_train3, y_train,
               weights=background_weight)
    model.save_weights(modelname+'.h5',overwrite=True)



