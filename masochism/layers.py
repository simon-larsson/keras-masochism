from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

from activations import crackmoid

class GroupAbnormalization(Layer):
    def __init__(self):
        raise NotImplementedError('Only a concept so far.')

class Crackmoid(Layer):
    """Crackmoid activation function.

       `f(x) = sigmoid(x)/x`
    """

    def __init__(self, **kwargs):
        super(Crackmoid, self).__init__(crackmoid, **kwargs)
        self.supports_masking = True
        
    def call(self, inputs, mask=None):
        return crackmoid(inputs)
    
    def get_config(self):
        base_config = super(Crackmoid, self).get_config()
        return base_config.items()
    
    def compute_output_shape(self, input_shape):
        return input_shape