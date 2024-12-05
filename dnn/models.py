"""Functions for DNN architectures"""

from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
import tensorflow as tf

from loss_func import *
from layers import res_conv3d_block, gating_signal, attention_block
from layers import enc_conv_block, dec_conv_block

def Attention_3DUnet(input_shape):
    tf.keras.backend.clear_session()  # clear the TF session and reset the parameters
    # input_shape=(None,None,1)  # layers.Reshape doesn't work with None shape
    n_filters = 128
    #input_shape=(32,32,32,1)
    #input_shape=(None,None,None,1)
    x = layers.Input(shape=input_shape)  # (None, y, x, 1)

    cv1_d = res_conv3d_block(x, kernel_size=3, n_filters=n_filters, dropout=0.1, batch_norm=True)  # (None, z, y, x, n_filters)
    p1_d = layers.AveragePooling3D(pool_size=(2, 2, 2))(cv1_d)  # (None, z/2, y/2, x/2, n_filters)

    cv2_d = res_conv3d_block(p1_d, kernel_size=3, n_filters=2*n_filters, dropout=0.1, batch_norm=True)  # (None, z/2, y/2, x/2, 2*n_filters)
    p2_d = layers.AveragePooling3D(pool_size=(2, 2, 2))(cv2_d)  # (None, z/4, y/4, x/4, 2*n_filters)

    cv3_d = res_conv3d_block(p2_d, kernel_size=3, n_filters=4*n_filters, dropout=0.1, batch_norm=True)  # (None, z/4, y/4, x/4, 4*n_filters)
    p3_d = layers.AveragePooling3D(pool_size=(2, 2, 2))(cv3_d)  # (None, z/8, y/8, x/8, 4*n_filters)

    cv4_d = res_conv3d_block(p3_d, kernel_size=3, n_filters=8*n_filters, dropout=0.1, batch_norm=True)  # (None, y/8, x/8, 8*n_filters)
    p4_d = layers.AveragePooling3D(pool_size=(2, 2, 2))(cv4_d)  # (None, y/16, x/16, 8*n_filters)

    cv5_d = res_conv3d_block(p4_d, kernel_size=3, n_filters=16*n_filters, dropout=0.1, batch_norm=True)  # (None, y/16, x/16, 16*n_filters)

    g4_u = gating_signal(cv5_d, 8*n_filters, batch_norm=True)
    att4_u = attention_block(cv4_d, g4_u, 8*n_filters)
    up4_u = layers.UpSampling3D(size=(2, 2, 2))(cv5_d)
    concat4_u = layers.concatenate([up4_u, att4_u], axis=-1)
    cv4_u = res_conv3d_block(concat4_u,  kernel_size=3, n_filters=8*n_filters, dropout=0.1, batch_norm=True)

    g3_u = gating_signal(cv4_u, 4*n_filters, batch_norm=True)
    att3_u = attention_block(cv3_d, g3_u, 4*n_filters)
    up3_u = layers.UpSampling3D(size=(2, 2, 2))(cv4_u)
    concat3_u = layers.concatenate([up3_u, att3_u], axis=-1)
    cv3_u = res_conv3d_block(concat3_u,  kernel_size=3, n_filters=4*n_filters, dropout=0.1, batch_norm=True)

    g2_u = gating_signal(cv3_u, 2*n_filters, batch_norm=True)
    att2_u = attention_block(cv2_d, g2_u, 2*n_filters)
    up2_u = layers.UpSampling3D(size=(2, 2, 2))(cv3_u)
    concat2_u = layers.concatenate([up2_u, att2_u], axis=-1)
    cv2_u = res_conv3d_block(concat2_u,  kernel_size=3, n_filters=2*n_filters, dropout=0.1, batch_norm=True)

    g1_u = gating_signal(cv2_u, n_filters, batch_norm=True)
    att1_u = attention_block(cv1_d, g1_u, n_filters)
    up1_u = layers.UpSampling3D(size=(2, 2, 2))(cv2_u)
    concat1_u = layers.concatenate([up1_u, att1_u], axis=-1)
    cv1_u = res_conv3d_block(concat1_u,  kernel_size=3, n_filters=n_filters, dropout=0.1, batch_norm=True)

    conv_final = layers.Conv3D(1, kernel_size=(1,1,1))(cv1_u) # RGB 이면 num_channels = 3
    conv_final = layers.BatchNormalization(axis=-1)(conv_final)
    y = layers.Activation('sigmoid')(conv_final)  # (None, n_zstacks, y, x, 1)

    model = models.Model(inputs=x, outputs=y)
    model.compile(loss=jaccard_loss, metrics=[jaccard_index, dice_coefficient], optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1))

    model.summary()

    return model



def reconstruct_2d_3d(input_shape):
    tf.keras.backend.clear_session()  # clear the TF session and reset the parameters
    n_filters = 128
    #x = layers.Input(shape=(1, 64, 64, 2))
    x = layers.Input(shape=input_shape)  # (None, 1, y, x, 2)

    cv1_d = enc_conv_block(x, n_filters=n_filters, dropout=0.1, batch_norm=True)  # (None, 1, y, x, n_filters)
    p1_d = layers.AveragePooling3D(pool_size=(1, 2, 2))(cv1_d)  # (None, 1, y/2, x/2, n_filters)

    cv2_d = enc_conv_block(p1_d, n_filters=2 * n_filters, dropout=0.1, batch_norm=True)  # (None, 1, y/2, x/2, 2*n_filters)
    p2_d = layers.AveragePooling3D(pool_size=(1, 2, 2))(cv2_d)  # (None, 1, y/4, x/4, 2*n_filters)

    cv3_d = enc_conv_block(p2_d, n_filters=4 * n_filters, dropout=0.1, batch_norm=True)  # (None, 1, y/4, x/4, 4*n_filters)
    p3_d = layers.AveragePooling3D(pool_size=(1, 2, 2))(cv3_d)  # (None, 1, y/8, x/8, 4*n_filters)

    cv4_d = enc_conv_block(p3_d, n_filters=8 * n_filters, dropout=0.1, batch_norm=True)  # (None, 1, y/8, x/8, 8*n_filters)
    p4_d = layers.AveragePooling3D(pool_size=(1, 2, 2))(cv4_d)  # (None, 1, y/16, x/16, 8*n_filters)

    cv5_d = enc_conv_block(p4_d, n_filters=16 * n_filters, dropout=0.1, batch_norm=True)  # (None, 1, y/16, x/16, 16*n_filters)

    dec = layers.Conv3DTranspose(8 * n_filters, (3, 3, 3), strides=(2, 1, 1), activation='relu', padding='same')(cv5_d)  # (None, 2, y/16, x/16, 8*n_filters)
    dec = layers.BatchNormalization(axis=-1)(dec)
    dec = layers.Conv3D(8 * n_filters, (2, 3, 3), activation='relu', padding='same')(dec)  # (None, 2, y/16, x/16, 8*n_filters)
    dec = layers.BatchNormalization(axis=-1)(dec)
    dec = layers.Dropout(0.1)(dec)

    dec = layers.Conv3DTranspose(8 * n_filters, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(dec)  # (None, 4, y/8, x/8, 8*n_filters)
    dec = layers.BatchNormalization(axis=-1)(dec)
    dec = layers.Conv3D(8 * n_filters, (3, 3, 3), activation='relu', padding='same')(dec)  # (None, 4, y/8, x/8, 8*n_filters)
    dec = layers.BatchNormalization(axis=-1)(dec)
    dec = layers.Dropout(0.1)(dec)

    dec = layers.Conv3DTranspose(4 * n_filters, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(dec)  # (None, 8, y/4, x/4, 4*n_filters)
    dec = layers.BatchNormalization(axis=-1)(dec)
    dec = layers.Conv3D(4 * n_filters, (3, 3, 3), activation='relu', padding='same')(dec)  # (None, 8, y/4, x/4, 4*n_filters)
    dec = layers.BatchNormalization(axis=-1)(dec)
    dec = layers.Dropout(0.1)(dec)

    dec = layers.Conv3DTranspose(2 * n_filters, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(dec)  # (None, 16, y/2, x/2, 2*n_filters)
    dec = layers.BatchNormalization(axis=-1)(dec)
    dec = layers.Conv3D(2 * n_filters, (3, 3, 3), activation='relu', padding='same')(dec)  # (None, 16, y/2, x/2, 2*n_filters)
    dec = layers.BatchNormalization(axis=-1)(dec)
    dec = layers.Dropout(0.1)(dec)

    dec = layers.Conv3DTranspose(n_filters, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(dec)  # (None, 32, y, x, n_filters)
    dec = layers.BatchNormalization(axis=-1)(dec)
    dec = layers.Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same')(dec)  # (None, 32, y, x, n_filters)
    dec = layers.BatchNormalization(axis=-1)(dec)
    dec = layers.Dropout(0.1)(dec)

    # x = dec_conv_block(x, n_filters=8 * n_filters, dropout=0.1, batch_norm=True)  # (None, 4, y/8, x/8, 8*n_filters)
    #
    # x = dec_conv_block(x, n_filters=4 * n_filters, dropout=0.1, batch_norm=True) # (None, 8, y/4, x/4, 4*n_filters)
    #
    # x = dec_conv_block(x, n_filters=2 * n_filters, dropout=0.1, batch_norm=True)  # (None, 16, y/2, x/2, 2*n_filters)
    #
    # x = dec_conv_block(x, n_filters=2 * n_filters, dropout=0.1, batch_norm=True)  # (None, 32, y, x, n_filters)

    conv_final = layers.Conv3D(1, kernel_size=(1, 1, 1))(dec)  # RGB 이면 num_channels = 3  # (None, 32, y, x, 1)
    conv_final = layers.BatchNormalization(axis=-1)(conv_final)
    y = layers.Activation('sigmoid')(conv_final)  # (None, 32, y, x, 1)

    model = models.Model(inputs=x, outputs=y)
    model.compile(loss=jaccard_loss, metrics=[jaccard_index, dice_coefficient], optimizer='adam')

    model.summary()

    return model


def Temporal_Conv1D_2D(duration, coor_dim=3, dimension=128):
    tf.keras.backend.clear_session()
    def causal_res_conv1d_block(input_layer, kernel_size, num_filters, dilation_rate):
        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(
            input_layer)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(
            x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)

        # if input_layer.shape[-1] != x.shape[-1]:
        shortcut = layers.Conv1D(filters=num_filters, kernel_size=1, padding='same')(input_layer)
        shortcut = layers.BatchNormalization(axis=-1)(shortcut)

        residual = layers.add([shortcut, x])

        return residual

    def res_conv2d_block(image, kernel_size, num_filters, dropout=0.1, batch_norm=True):
        conv = layers.Conv2D(num_filters, (kernel_size, kernel_size), padding='same', strides=1)(image)
        # padding same: input shape = output shape with zero padding (only when stride = 1)
        # stride >=2 이면 output shape = input shape / stride
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=-1)(
                conv)  # input = (num_of_images, height, width, channels)에서 channels 방향으로 normalize
        conv = layers.Activation('relu')(conv)

        conv = layers.Conv2D(num_filters, (kernel_size, kernel_size), padding='same', strides=1)(conv)
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=-1)(
                conv)  # input = (num_of_images, height, width, channels)에서 channels 방향으로 normalize
        conv = layers.Activation('relu')(conv)

        if dropout > 0:
            conv = layers.Dropout(dropout)(conv)

        shortcut = layers.Conv2D(num_filters, (1, 1), padding='same', strides=1)(image)
        if batch_norm is True:
            shortcut = layers.BatchNormalization(axis=-1)(shortcut)

        residual = layers.add([shortcut, conv])

        return residual

    assert coor_dim >= 2, "coordinate dimension must be larger than 1"
    if coor_dim < 3:
        kernel_size_2d = 2
    elif coor_dim >= 3:
        kernel_size_2d = 3

    x = layers.Input(shape=(duration, coor_dim))

    x_2d = layers.Reshape(target_shape=(x.shape[1], x.shape[2], 1))(x)
    cv1_2d = res_conv2d_block(x_2d, kernel_size=kernel_size_2d, num_filters=16, dropout=0.1, batch_norm=True)
    cv2_2d = res_conv2d_block(cv1_2d, kernel_size=kernel_size_2d, num_filters=32, dropout=0.1, batch_norm=True)
    f_2d = layers.Flatten()(cv2_2d)
    drop_2d = layers.Dropout(0.1)(f_2d)
    d1_2d = layers.Dense(100, activation='relu')(drop_2d)

    cv1 = causal_res_conv1d_block(x, kernel_size=3, num_filters=16, dilation_rate=1)
    p1 = layers.MaxPooling1D(pool_size=2)(cv1)
    cv2 = causal_res_conv1d_block(p1, kernel_size=3, num_filters=32, dilation_rate=2)
    p2 = layers.MaxPooling1D(pool_size=2)(cv2)
    cv3 = causal_res_conv1d_block(p2, kernel_size=3, num_filters=64, dilation_rate=4)
    f = layers.Flatten()(cv3)
    drop = layers.Dropout(0.1)(f)
    d1 = layers.Dense(100, activation='relu')(drop)

    concat = layers.Concatenate(name='Concatenate')([d1_2d, d1])
    concat_d1 = layers.Dense(dimension, activation='relu')(concat)
    concat_do1 = layers.Dropout(rate=0.1)(concat_d1)
    concat_d2 = layers.Dense(cv3.shape[1] * cv3.shape[2], activation='relu')(concat_do1)
    concat_do2 = layers.Dropout(rate=0.1)(concat_d2)
    concat_rs = layers.Reshape(target_shape=(cv3.shape[1], cv3.shape[2]))(concat_do2)

    uc1 = causal_res_conv1d_block(concat_rs, kernel_size=3, num_filters=64, dilation_rate=4)
    us1 = layers.UpSampling1D(size=2)(uc1)
    uc2 = causal_res_conv1d_block(us1, kernel_size=3, num_filters=32, dilation_rate=2)
    us2 = layers.UpSampling1D(size=2)(uc2)
    uc3 = causal_res_conv1d_block(us2, kernel_size=3, num_filters=16, dilation_rate=1)
    y = layers.Conv1D(filters=coor_dim, kernel_size=3, padding='same', activation=None)(uc3)


    model = models.Model(x, y)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1), metrics=['accuracy'])
    model.summary()

    return model

