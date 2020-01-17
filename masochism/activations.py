import keras.backend as K
from keras.layers import Activation
from keras.utils import get_custom_objects

def crackmoid(x):
    """Crackmoid activation function.

    # Arguments
        x: Input tensor.

    # Returns
        The Crackmoid activation: `sigmoid(x)/x`.
    """

    return K.sigmoid(x)/(x + K.epsilon())

get_custom_objects().update({'crackmoid': Activation(crackmoid)})