"""Functions for loss functions"""

from tensorflow.keras import backend as K
import tensorflow as tf

def dice_coefficient(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1]) # shape을 [-1]로 두면 tensor -> 1d vector(a, b, c, d) -> (a*b*c*d, )
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
    # y_true, y_pred = 0 or 1 -> math.reduce_sum increment only when both y_true and y_pred are 1
    dice_coef = (2.0 * intersection + 1e-10)/ (tf.math.reduce_sum(y_true_f)+tf.math.reduce_sum(y_pred_f) + 1e-10)
    # sum of elements across dimension (dimension specify 안하면 모든 축에 대해 더해서 결국 상수값 하나가 나옴)
    # 분자 분모에 1e-10을 더하는건 0으로 나누는걸 방지하기 위해
    return dice_coef

def dice_loss(y_true, y_pred):
    return 1- dice_coefficient(y_true, y_pred)


def jaccard_index(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1]) # shape을 [-1]로 두면 tensor -> 1d vector(a, b, c, d) -> (a*b*c*d, )
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
    # y_true, y_pred = 0 or 1 -> math.reduce_sum increment only when both y_true and y_pred are 1
    jaccard_idx = (intersection + 1e-10)/ (tf.math.reduce_sum(y_true_f)+tf.math.reduce_sum(y_pred_f) - intersection + 1e-10)
    # sum of elements across dimension (dimension specify 안하면 모든 축에 대해 더해서 결국 값 하나가 나옴)
    # 분자 분모에 1e-10을 더하는건 분모가 0이 되어 error 뜨는걸 방지하기 위해
    return jaccard_idx

def jaccard_loss(y_true, y_pred):
    return 1-jaccard_index(y_true, y_pred)

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed