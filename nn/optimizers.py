import numpy as np


class Optimizer:
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.lr / (1 + self.decay * self.iterations)

    def post_update_params(self):
        self.iterations += 1


class SGD(Optimizer):
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.lr = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def update_params(self, layer):
        if not hasattr(layer, 'W_momentums'):
            layer.W_momentums = np.zeros_like(layer.W)
            layer.b_momentums = np.zeros_like(layer.b)

        layer.W_momentums = self.momentum * layer.W_momentums - self.current_lr * layer.dW
        layer.b_momentums = self.momentum * layer.b_momentums - self.current_lr * layer.db

        layer.W += layer.W_momentums
        layer.b += layer.b_momentums


class Adagrad(Optimizer):
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.lr = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def update_params(self, layer):
        if not hasattr(layer, 'W_cache'):
            layer.W_cache = np.zeros_like(layer.W)
            layer.b_cache = np.zeros_like(layer.b)

        layer.W_cache = layer.W_cache + layer.dW**2
        layer.b_cache = layer.b_cache + layer.db**2

        layer.W += -self.current_lr * (layer.dW / np.sqrt(layer.W_cache + self.epsilon))
        layer.b += -self.current_lr * (layer.db / np.sqrt(layer.b_cache + self.epsilon))


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, decay=0., rho=0.9, epsilon=1e-7):
        self.lr = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.rho = rho
        self.epsilon = epsilon

    def update_params(self, layer):
        if not hasattr(layer, 'W_cache'):
            layer.W_cache = np.zeros_like(layer.W)
            layer.b_cache = np.zeros_like(layer.b)

        layer.W_cache = self.rho * layer.W_cache + (1 - self.rho) * layer.dW**2
        layer.b_cache = self.rho * layer.b_cache + (1 - self.rho) * layer.db**2

        layer.W += -self.current_lr * (layer.dW / np.sqrt(layer.W_cache + self.epsilon))
        layer.b += -self.current_lr * (layer.db / np.sqrt(layer.b_cache + self.epsilon))


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.lr = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layer):
        if not hasattr(layer, 'W_momentums'):
            layer.W_momentums = np.zeros_like(layer.W)
            layer.b_momentums = np.zeros_like(layer.b)
            layer.W_cache = np.zeros_like(layer.W)
            layer.b_cache = np.zeros_like(layer.b)

        layer.W_momentums = self.beta_1 * layer.W_momentums + (1 - self.beta_1) * layer.dW
        layer.b_momentums = self.beta_1 * layer.b_momentums + (1 - self.beta_1) * layer.db

        layer.W_cache = self.beta_2 * layer.W_cache + (1 - self.beta_2) * layer.dW**2
        layer.b_cache = self.beta_2 * layer.b_cache + (1 - self.beta_2) * layer.db**2

        W_momentum_corrected = layer.W_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        b_momentums_corrected = layer.b_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        W_cache_corrected = layer.W_cache / (1 - self.beta_2 ** (self.iterations + 1))
        b_cache_corrected = layer.b_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.W += -self.current_lr * (W_momentum_corrected / np.sqrt(W_cache_corrected + self.epsilon))
        layer.b += -self.current_lr * (b_momentums_corrected / np.sqrt(b_cache_corrected + self.epsilon))
