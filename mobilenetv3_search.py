# -*- coding: utf-8 -*-
"""Vitis-AI_compatible_MobileNetV3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HmW1Jjbxo1seBlTEiwBFI4e1rPv-Rizh

##MobileNetV3
"""

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input, Reshape, Add
from keras.layers import Conv2D, MaxPooling2D, Multiply, ReLU
from keras.layers import DepthwiseConv2D, ReLU, GlobalAveragePooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator

"""###Blocks"""

def H_Swish(inputs):
    #x = inputs+3
    x = ReLU(max_value=6)(inputs) #x = x*ReLU(max_value=6)(inputs+3)/6
    #x = x / 6
    #x = Multiply()([inputs, x])
    return x

def se_block(inputs, filters):
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(filters // 4)(x)
    x = ReLU(max_value=6)(x)
    x = Dropout(0.5)(x)
    x = Dense(filters)(x)
    x = H_Swish(x)
    x = Reshape((1, 1, filters))(x)
    #un-comment the line below will go boom
    x = Multiply()([x, inputs])
    return x

def bottleneck(inputs, filters, kernel_size, alpha, stride=1, expansion=1, se=False):
    filters = int(filters * alpha)
    expanded_filters = expansion * filters

    if stride > 1:
        inputs = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(inputs)
        inputs = BatchNormalization()(inputs)
        x = H_Swish(inputs)

    x = Conv2D(expanded_filters, kernel_size=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = H_Swish(x)

    x = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = H_Swish(x)

    if se:
        x = se_block(x,expanded_filters)

    x = Conv2D(filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    if stride > 1:
        inputs = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(inputs)
        inputs = BatchNormalization()(inputs)
    if inputs.shape[-1] == filters: #if inputs.shape == x.shape:
        x = Add()([inputs, x])

    x = H_Swish(x)
    return x

"""###Datasets"""

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Data preprocessing
x_train = x_train.reshape((50000,32,32,3)).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train)

x_test = x_test.reshape((10000,32,32,3)).astype('float32') / 255
y_test = keras.utils.to_categorical(y_test)

"""###Search"""

!pip install keras-tuner -q
import keras_tuner

def search_cnn(hp):

  inputs = keras.Input(shape=(32,32,3))

  first_filters = int(16 * 1.0)

  x = Conv2D(first_filters, kernel_size=3, strides=(1, 1), padding='same')(inputs)
  x = BatchNormalization()(x)
  x = ReLU(max_value=6)(x)

  # MobileNetV3 bottleneck
  #1 bneck
  filters1 = hp.Int("filters1", min_value=1, max_value=112, step=3)
  stride1 = hp.Int("stride1", min_value=1, max_value=2, step=1)
  expansion1 = hp.Int("expansion1", min_value=1, max_value=10, step=1)
  se1 = hp.Boolean("se1")
  x = bottleneck(x, filters1, (3, 3), 1.0, stride=stride1, expansion=expansion1, se=se1)
  #2 bneck
  filters2 = hp.Int("filters2", min_value=1, max_value=112, step=3)
  stride2 = hp.Int("stride2", min_value=1, max_value=2, step=1)
  expansion2 = hp.Int("expansion2", min_value=1, max_value=10, step=1)
  se2 = hp.Boolean("se2")
  x = bottleneck(x, filters2, (3, 3), 1.0, stride=stride2, expansion=expansion2, se=se2)
  #3 bneck
  filters3 = hp.Int("filters3", min_value=1, max_value=112, step=3)
  stride3 = hp.Int("stride3", min_value=1, max_value=2, step=1)
  expansion3 = hp.Int("expansion3", min_value=1, max_value=10, step=1)
  se3 = hp.Boolean("se3")
  x = bottleneck(x, filters3, (3, 3), 1.0, stride=stride3, expansion=expansion3, se=se3)
  #4 bneck
  filters4 = hp.Int("filters4", min_value=1, max_value=112, step=3)
  stride4 = hp.Int("stride4", min_value=1, max_value=2, step=1)
  expansion4 = hp.Int("expansion4", min_value=1, max_value=10, step=1)
  se4 = hp.Boolean("se4")
  x = bottleneck(x, filters4, (3, 3), 1.0, stride=stride4, expansion=expansion4, se=se4)
  #5 bneck
  filters5 = hp.Int("filters5", min_value=1, max_value=112, step=3)
  stride5 = hp.Int("stride5", min_value=1, max_value=2, step=1)
  expansion5 = hp.Int("expansion5", min_value=1, max_value=10, step=1)
  se5 = hp.Boolean("se5")
  x = bottleneck(x, filters5, (3, 3), 1.0, stride=stride5, expansion=expansion5, se=se5)

  # output layer
  convfilter = hp.Int("convfilter", min_value=16, max_value=128, step=4)
  x = Conv2D(convfilter, kernel_size=1, strides=(1, 1), padding='same')(x)   #576
  x = BatchNormalization()(x)
  x = ReLU(max_value=6)(x)
  x = GlobalAveragePooling2D()(x)
  x = Flatten()(x)
  outputs = Dense(10, activation='softmax')(x)

  model = keras.Model(inputs, outputs)
  #model.summary()
  model.compile(optimizer='rmsprop',  #'rmsprop'
            loss="categorical_crossentropy",
            metrics=['accuracy']
            )
  return model
search_cnn(keras_tuner.HyperParameters())

tuner = keras_tuner.RandomSearch(
    hypermodel=search_cnn,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)
tuner.search_space_summary()

tuner.search(x_train, y_train, epochs=6, validation_data=(x_test, y_test))

tuner.results_summary()

models = tuner.get_best_models(num_models=1)
best_model = models[0]
best_model.summary()