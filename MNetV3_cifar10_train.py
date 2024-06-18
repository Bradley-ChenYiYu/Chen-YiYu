'''
Create & train a custom cnn model for CIFAR-10 classification
Tensorflow 2.3

Author: chao.zhang, YiYu_Chen
'''

import os

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from load_data import get_cifar10_dataset
from keras.preprocessing.image import ImageDataGenerator

from keras import activations
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input, Reshape, Add
from keras.layers import Conv2D, MaxPooling2D, Multiply, ReLU
from keras.layers import DepthwiseConv2D, ReLU, GlobalAveragePooling2D, ZeroPadding2D

MODEL_DIR = './models'
FLOAT_MODEL = 'float_model.h5'

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

def H_Swish(inputs):    
    #x = inputs+3
    #c3 = 3*tf.ones(tf.shape(inputs))
    #x = x = Add()([inputs, c3])
    x = ReLU(max_value=6)(inputs) #x = x*ReLU(max_value=6)(inputs+3)/6
    #x = x / 6
    #c1_6 = tf.ones(tf.shape(x))/6
    #x = Multiply()([x, c1_6])
    return x

def se_block(inputs, filters):
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(filters // 4)(x)
    x = ReLU(max_value=6)(x)
    x = Dropout(0.5)(x)
    x = Dense(filters)(x)
    x = H_Swish(x)
    x = Reshape((1, 1, filters))(x)
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

    if stride == 2:
      x = ZeroPadding2D(padding=(1,1))(x)
    
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
    if stride == 1 and inputs.shape[-1] == filters: #if inputs.shape == x.shape:
        x = Add()([inputs, x])

    x = H_Swish(x)
    return x

def customcnn():    #try dropout
    inputs = keras.Input(shape=(32,32,3))


    first_filters = int(16 * 1.0)

    x = Conv2D(first_filters, kernel_size=3, strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)

    # MobileNetV3 bottleneck
    x = bottleneck(x, 37, (3, 3), 1.0, stride=2, expansion=10, se=True)    #True
    x = bottleneck(x, 67, (3, 3), 1.0, stride=1, expansion=1, se=False)    #False
    x = bottleneck(x, 55, (3, 3), 1.0, stride=1, expansion=7, se=True)    #True
    x = bottleneck(x, 73, (3, 3), 1.0, stride=1, expansion=2, se=True)   #False
    x = bottleneck(x, 43, (3, 3), 1.0, stride=1, expansion=1, se=False)   #True

    # output layer
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.summary()
    return model

print("\nLoad cifar10 dataset..")
(train_dataset, val_dataset, test_dataset) = get_cifar10_dataset()

# build cnn model
print("\nCreate custom cnn..")
model = customcnn()

optimizer = keras.optimizers.SGD(
                                momentum=0.5,
                                learning_rate=1e-1
                                )
model.compile(optimizer=optimizer,  #'rmsprop' 
            loss="categorical_crossentropy",
            metrics=['accuracy']
            )

# Train the model for 10 epochs using a dataset
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
print("\nFit on dataset..")

history = model.fit(train_dataset, epochs=200,validation_data=val_dataset, 
                    callbacks=[reduce_lr, early_stopping]
                    ) #epochs=300

# Save model
path = os.path.join(MODEL_DIR, FLOAT_MODEL)
print("\nSave trained model to{}.".format(path))
model.save(path)

# Evaluate model with test data
print("\nEvaluate model on test dataset..")
loss, acc = model.evaluate(test_dataset)  # returns loss and metrics
print("loss: %.3f" % loss)
print("acc: %.3f" % acc)


