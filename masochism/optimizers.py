from keras.legacy import interfaces
from keras.optimizers import Optimizer
import keras.backend as K

class SGDIM(Optimizer):
    """Stochastic gradual descent into madness optimizer.

    Like SGD but sometimes takes a step in the wrong direction.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    
    # Arguments
        learning_rate: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        nesterov: boolean. Whether to apply Nesterov momentum.
        madness: Likelihood of taking a misstep.
    """

    def __init__(self, 
                 learning_rate=0.01, 
                 momentum=0.,
                 nesterov=False,
                 madness=0.1,
                 **kwargs):
        
        learning_rate = kwargs.pop('lr', learning_rate)
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.initial_madness = madness
        
        super(SGDIM, self).__init__(**kwargs)
        
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(self.initial_decay, name='decay')
            
            if madness <= 0:
                raise ValueError('"In a mad world, only the mad are sane". Value of '
                                 '`madness` has to be over zero.')
            self.madness = K.variable(madness, name='madness')
    
        self.nesterov = nesterov

    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        # misstep condition based on madness
        misstep = K.switch(K.less(K.random_uniform([]), self.madness), 
                           lambda: K.constant(-1), 
                           lambda: K.constant(1))
        
        lr = misstep*self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape, name='moment_' + str(i))
                   for (i, shape) in enumerate(shapes)]
        
        self.weights = [self.iterations] + moments
        
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'madness': float(K.get_value(self.madness))}
        base_config = super(SGDIM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
