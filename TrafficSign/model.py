from tensorflow.keras.layers import Input, Concatenate, Conv2D, Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.initializers import glorot_normal
import tensorflow.keras.backend as K
from tensorflow.keras.activations import swish
from tensorflow.keras import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam

init = glorot_normal()

def block(inputs, filters = 64, block_name='block'):
    with K.name_scope(block_name):
        conv1 = Conv2D(filters, 3, 1, padding='same', kernel_initializer=init)(inputs)
        bn1 = BatchNormalization()(conv1)
        act1 = swish(bn1)
        dp1 = Dropout(0.5)(act1)

        conv2 = Conv2D(filters, 5, 1, padding='same', kernel_initializer=init)(inputs)
        bn2 = BatchNormalization()(conv2)
        act2 = swish(bn2)
        dp2 = Dropout(0.5)(act2)

        conv3 = Conv2D(filters, 1, 1, padding='same', kernel_initializer=init)(inputs)
        bn3 = BatchNormalization()(conv3)
        act3 = swish(bn3)
        dp3 = Dropout(0.5)(act3)

        return Concatenate(axis=1)([dp1, dp2, dp3])

def conv_block1(inputs, filters, block_name='block'):
    with K.name_scope(block_name):
        conv1 = Conv2D(filters, 3, 2, padding='same', kernel_initializer=init)(inputs)
        bn1 = BatchNormalization()(conv1)
        act1 = swish(bn1)

        conv2 = Conv2D(filters, 3, 1, padding='same', kernel_initializer=init)(act1)
        bn2 = BatchNormalization()(conv2)
        act1 = swish(bn2)

        return Conv2D(filters, 3, 2, padding='same', kernel_initializer=init)(inputs) + act1


def get_model(inputs=40):
    img_inputs = Input(shape=(inputs, inputs, 3))

    cv1 = Conv2D(32, 3, 1, padding='valid', activation='relu', kernel_initializer=init)(img_inputs)
    x = Conv2D(64, 3, 2, padding='valid', activation='relu', kernel_initializer=init)(cv1)

    for i in range(5):
        x = block(x, 128, 'block{}'.format(i))

    for i in range(5, 7):
        x = conv_block1(x, 256, 'block{}'.format(i))

    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_initializer=init)(x)
    x = Dropout(0.5)(x)
    x = Dense(43, activation='softmax')(x)

    model = Model(inputs=img_inputs, outputs=x, name='traffic_model')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    return model

