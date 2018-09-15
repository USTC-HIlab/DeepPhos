from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers import Input, merge, Flatten
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from keras.layers import Conv1D,Conv2D, MaxPooling2D


def conv_factory(x, init_form, nb_filter, filter_size_block, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout

    :param x: Input keras network
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras network with b_norm, relu and convolution2d added
    :rtype: keras network
    """
    #x = Activation('relu')(x)
    x = Conv1D(nb_filter, filter_size_block,
                      init=init_form,
                      activation='relu',
                      border_mode='same',
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, init_form, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D

    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool

    """
    #x = Activation('relu')(x)
    x = Conv1D(nb_filter, 1,
                      init=init_form,
                      activation='relu',
                      border_mode='same',
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    #x = AveragePooling2D((2, 2),padding='same')(x)
    x = AveragePooling1D(pool_size=2, padding='same')(x)

    return x


def denseblock(x, init_form, nb_layers, nb_filter, growth_rate,filter_size_block,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones

    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """

    list_feat = [x]
    concat_axis = -1

    for i in range(nb_layers):
        x = conv_factory(x, init_form, growth_rate, filter_size_block, dropout_rate, weight_decay)
        list_feat.append(x)
        x = merge(list_feat, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate
    return x

def Phos(nb_classes, nb_layers,img_dim1,img_dim2,img_dim3, init_form, nb_dense_block,
             growth_rate,filter_size_block1,filter_size_block2,filter_size_block3,
             nb_filter, filter_size_ori,
             dense_number,dropout_rate,dropout_dense,weight_decay):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param nb_layers:int --numbers of layers in a dense block
    :param filter_size_ori: int -- filter size of first conv1d
    :param dropout_dense: float---drop out rate of dense

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """
    # first input of 33 seq #
    main_input = Input(shape=img_dim1)
    #model_input = Input(shape=img_dim)
    # Initial convolution
    x1 = Conv1D(nb_filter, filter_size_ori,
                      init = init_form,
                      activation='relu',
                      border_mode='same',
                      bias=False,
                      W_regularizer=l2(weight_decay))(main_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x1 = transition(x1, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)

    x1 = Activation('relu',name='seq1')(x1)

    # second input of 21 seq #
    input2 = Input(shape=img_dim2)
    x2 = Conv1D(nb_filter, filter_size_ori,
                init=init_form,
                activation='relu',
                border_mode='same',
                bias=False,
                W_regularizer=l2(weight_decay))(input2)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
        # add transition
        x2 = transition(x2, init_form, nb_filter, dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                    dropout_rate=dropout_rate,
                    weight_decay=weight_decay)

    x2 = Activation('relu')(x2)

    #third input seq of 15 #
    input3 = Input(shape=img_dim3)
    x3 = Conv1D(nb_filter, filter_size_ori,
                init=init_form,
                activation='relu',
                border_mode='same',
                bias=False,
                W_regularizer=l2(weight_decay))(input3)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
        # add transition
        x3 = transition(x3, init_form, nb_filter, dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                    dropout_rate=dropout_rate,
                    weight_decay=weight_decay)

    x3 = Activation('relu')(x3)

    # contact 3 output features #
    x = merge([x1, x2, x3], mode='concat', concat_axis=-2, name='contact_multi_seq')

    #x = GlobalAveragePooling1D()(x)

    x = Flatten()(x)

    x = Dense(dense_number,
              name ='Dense_1',
              activation='relu',init = init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)

    x = Dropout(dropout_dense)(x)
    #softmax
    x = Dense(nb_classes,
              name = 'Dense_softmax',
              activation='softmax',init = init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)

    phos_model = Model(input=[main_input,input2,input3], output=[x], name="multi-DenseNet")
    #feauture_dense = Model(input=[main_input, input2, input3], output=[x], name="multi-DenseNet")

    return phos_model
