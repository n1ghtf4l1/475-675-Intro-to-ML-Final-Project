"""Functions for DNN layers"""

from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
import tensorflow as tf

def res_conv3d_block(inp, kernel_size, n_filters, dropout=0.1, batch_norm=True):
    ''' Apply Residual 2D convolution with dropout, relu and batch norm
        (None, 32, 32, 32, 1) -> (None, 32, 32, 32, n_filters)
    Parameters:
    ----------
    inp: np.array()
        input tensor, shape = (None, z, row, col, 1)
    kernel_size: int
        apply (kernel_size, kernel_size) convolution
    n_filters: int
        determine amount of convolution filters to be generated
    dropout: float
        rate of dropout
    batch_norm: bool
        when True apply batch norm on channel axis

    Returns:
    -------
    residual: np.array()
        output tensor, shape = (None, z, row, col, n_filters)

    '''

    conv = layers.Conv3D(n_filters, (kernel_size, kernel_size, kernel_size), padding='same', strides=1)(inp)
    # padding same: input shape = output shape with zero padding (only when stride = 1)
    # stride >=2 이면 output shape = input shape / stride
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=-1)(conv)
        # input = (num_of_images, height, width, channels), normalize by channel direction
    conv = layers.Activation('relu')(conv)

    conv = layers.Conv3D(n_filters, (kernel_size, kernel_size, kernel_size), padding='same', strides=1)(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=-1)(conv)
        # input = (num_of_images, height, width, channels), normalize by channel direction
    conv = layers.Activation('relu')(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv3D(n_filters, (1, 1, 1), padding='same', strides=1)(inp)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=-1)(shortcut)

    residual = layers.add([shortcut, conv])

    return residual

def repeat_tensor(tensor, rep):
    # lambda inputs(x = tensor, rep_num = rep): (None, z, y, x, c) -> (None, z, y, x, rep*c)
    # Lambda layer(input = function 형태여서 lambda 함수 사용)
    # 또 Lambda layer는 하나의 argument만 input으로 두는데 argument 하나 더 필요하면 arguments = {'y':y} 사용

    return layers.Lambda(lambda x, rep_num: K.repeat_elements(x, rep_num, axis=-1), arguments={'rep_num': rep})(tensor)

def gating_signal(x, n_filters, batch_norm=True):
    # input: (None, z, y, x, n) -> output (None, z, y, x, n_filters)
    conv = layers.Conv3D(n_filters, kernel_size=(1,1,1), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=-1)(conv)
    conv = layers.Activation('relu')(conv)
    return conv

def attention_block(x, g, n_filters):
    # x: (None, z, y, x, c), g: (None, z/2, y/2, x/2, c) -> output: (None, z, y, x, c)

    shape_x = K.int_shape(x)  # (None, z, y, x, c)
    shape_g = K.int_shape(g)  # (None, z/2, y/2, x/2, c)

    theta_x = layers.Conv3D(n_filters, kernel_size=(2,2,2), strides=(2,2,2), padding='same')(x)  # (None,z/2, y/2, x/2, c)
    shape_theta_x = K.int_shape(theta_x)  # (None,z/2, y/2, x/2, c)

    phi_g = layers.Conv3D(n_filters, kernel_size=(1,1,1), padding='same')(g)  # (None,z/2, y/2, x/2, c)
    upsample_g = layers.Conv3DTranspose(n_filters, kernel_size=(3,3,3), strides=(1,1,1), padding='same')(phi_g)  # (None, z/2, y/2, x/2, c)
                                        #strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2], shape_theta_x[3] // shape_g[3]),

    concat_xg = layers.add([upsample_g, theta_x])  # (None,z/2, y/2, x/2, c)
    act_xg = layers.Activation('relu')(concat_xg) # (None,z/2, y/2, x/2, c)
    psi = layers.Conv3D(1, kernel_size=(1,1,1), padding='same')(act_xg)  # (None,z/2, y/2, x/2, 1)
    sigmoid_xg = layers.Activation('sigmoid')(psi)  # (None,z/2, y/2, x/2, 1)
    shape_sigmoid = K.int_shape(sigmoid_xg)  # (None,z/2, y/2, x/2, 1)
    upsample_psi = layers.UpSampling3D(size=(2,2,2))(sigmoid_xg) # (None, z, y, x, 1)
        # size = ( shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3] )

    upsample_psi = repeat_tensor(upsample_psi, shape_x[-1]) # (None, z, y, x, c)

    y = layers.multiply([upsample_psi, x])  # element-wise multiplication, (None, z, y, x, c)

    result = layers.Conv3D(shape_x[-1], kernel_size=(1,1,1), padding='same')(y) # (None, z, y, x, c)
    result = layers.BatchNormalization(axis=-1)(result) # (None, z, y, x, c)
    return result



def enc_conv_block(inp, n_filters, dropout=0.1, batch_norm=True):
    ''' Apply Residual convolution with dropout, relu and batch norm
        (None, 1, 32, 32, 1) -> (None, 1, 32, 32, n_filters)
    Parameters:
    ----------
    inp: np.array()
        input tensor, shape = (None, z, row, col, 1)
    kernel_size: int
        apply (kernel_size, kernel_size) convolution
    n_filters: int
        determine amount of convolution filters to be generated
    dropout: float
        rate of dropout
    batch_norm: bool
        when True apply batch norm on channel axis

    Returns:
    -------
    residual: np.array()
        output tensor, shape = (None, z, row, col, n_filters)

    '''

    conv = layers.Conv3D(n_filters, (1, 3, 3), padding='same', strides=1)(inp) # (None, z, y, x, n_filters)
    # padding same: input shape = output shape with zero padding (only when stride = 1)
    # stride >=2, then output shape = input shape / stride
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=-1)(conv)
        # input = (num_of_images, height, width, channels), normalize by channel direction
    conv = layers.Activation('relu')(conv)  # (None, z, y, x, n_filters)

    conv = layers.Conv3D(n_filters, (1, 3, 3), padding='same', strides=1)(conv)  # (None, z, y, x, n_filters)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=-1)(conv)
        # input = (num_of_images, height, width, channels), normalize by channel direction
    conv = layers.Activation('relu')(conv)  # (None, z, y, x, n_filters)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)  # (None, z, y, x, n_filters)

    shortcut = layers.Conv3D(n_filters, (1, 1, 1), padding='same', strides=1)(inp)  # (None, z, y, x, n_filters)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=-1)(shortcut)  # (None, z, y, x, n_filters)

    residual = layers.add([shortcut, conv])  # (None, z, y, x, n_filters)

    return residual


def dec_conv_block(inp, n_filters, dropout=0.1, batch_norm=True):
    ''' Apply transpose convolution with dropout, relu and batch norm
        (None, z, y, x, 1) -> (None, 2z, 2y, 2x, n_filters)
    Parameters:
    ----------
    inp: np.array()
        input tensor, shape = (None, z, y, x, 1)
    kernel_size: int
        apply (kernel_size, kernel_size) convolution
    n_filters: int
        determine amount of convolution filters to be generated
    dropout: float
        rate of dropout
    batch_norm: bool
        when True apply batch norm on channel axis

    Returns:
    -------
    x: np.array()
        output tensor, shape = (None, 2z, 2y, 2x, n_filters)

    '''
    x = layers.Conv3DTranspose(8 * n_filters, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(inp)
    if batch_norm is True:
        x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv3D(8 * n_filters, (3, 3, 3), activation='relu', padding='same')(x)
    if batch_norm is True:
        x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(dropout)(x)

    return x