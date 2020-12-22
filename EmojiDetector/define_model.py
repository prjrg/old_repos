from keras import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D
from keras.activations import relu
import keras.backend as K
from keras.initializers import  glorot_normal


init = glorot_normal()

def conv_block(inputs, filters = 64, block_name='block'):
    with K.name_scope(block_name):
        conv1 = Conv2D(filters, 3, 1, padding='same', kernel_initializer=init)(inputs)
        bn1 = BatchNormalization()(conv1)
        act1 = relu(bn1)
        dp1 = Dropout(0.5)(act1)

        conv2 = Conv2D(filters, 5, 1, padding='same', kernel_initializer=init)(inputs)
        bn2 = BatchNormalization()(conv2)
        act2 = relu(bn2)
        dp2 = Dropout(0.5)(act2)

        conv3 = Conv2D(filters, 1, 1, padding='same', kernel_initializer=init)(inputs)
        bn3 = BatchNormalization()(conv3)
        act3 = relu(bn3)
        dp3 = Dropout(0.5)(act3)

        return Concatenate(axis=1)([dp1, dp2, dp3])


def conv_block1(inputs, filters, block_name='block'):
    with K.name_scope(block_name):
        conv1 = Conv2D(filters, 3, 2, padding='same', kernel_initializer=init)(inputs)
        bn1 = BatchNormalization()(conv1)
        act1 = relu(bn1)

        conv2 = Conv2D(filters, 3, 1, padding='same', kernel_initializer=init)(act1)
        bn2 = BatchNormalization()(conv2)
        act1 = relu(bn2)

        return Conv2D(filters, 3, 2, padding='same', kernel_initializer=init)(inputs) + act1


def define_network(inputs=48):
    img_inputs = Input(shape=(inputs, inputs, 1))

    cv1 = Conv2D(32, 3, 1, padding='valid', activation='relu', kernel_initializer=init)(img_inputs)
    cv2 = Conv2D(64, 3, 1, padding='valid', activation='relu', kernel_initializer=init)(cv1)
    x = MaxPooling2D(2)(cv2)

    for i in range(5):
        x = conv_block(x, 128, 'block{}'.format(i))

    for i in range(5, 8):
        x = conv_block1(x, 256, 'block{}'.format(i))

    x = Conv2D(128, 3, strides=1, padding='valid', activation='relu', kernel_initializer=init)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_initializer=init)(x)
    x = Dropout(0.5)(x)
    x = Dense(7, activation='softmax')(x)

    return Model(inputs=img_inputs, outputs=x, name='emoji_model')

