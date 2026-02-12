import numpy as np


class Conv2D:
    def __init__(self, n_filters, kernel_size, n_channels, stride=1, padding=0):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.stride = stride
        self.padding = padding

        # He initialization
        self.W = np.random.randn(n_filters, n_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (n_channels * kernel_size * kernel_size))
        self.b = np.zeros((n_filters, 1))

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, _, h_in, w_in = inputs.shape

        if self.padding > 0:
            self.inputs_padded = np.pad(
                inputs,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            self.inputs_padded = inputs

        h_out = (h_in + 2*self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2*self.padding - self.kernel_size) // self.stride + 1

        self.output = np.zeros((batch_size, self.n_filters, h_out, w_out))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                receptive_field = self.inputs_padded[:, :, h_start:h_end, w_start:w_end]

                for f in range(self.n_filters):
                    self.output[:, f, i, j] = np.sum(
                        receptive_field * self.W[f, :, :, :],
                        axis=(1, 2, 3)
                    ) + self.b[f]

    def backward(self, dvalues):
        batch_size, _, h_out, w_out = dvalues.shape

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.dinputs_padded = np.zeros_like(self.inputs_padded)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                receptive_field = self.inputs_padded[:, :, h_start:h_end, w_start:w_end]

                for f in range(self.n_filters):
                    self.dW[f] += np.sum(
                        receptive_field * dvalues[:, f, i, j][:, None, None, None],
                        axis=0
                    )

                    self.db[f] += np.sum(dvalues[:, f, i, j])

                    self.dinputs_padded[:, :, h_start:h_end, w_start:w_end] += (
                        self.W[f] * dvalues[:, f, i, j][:, None, None, None]
                    )

        # Remove padding
        if self.padding > 0:
            self.dinputs = self.dinputs_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            self.dinputs = self.dinputs_padded


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, n_channels, h_in, w_in = inputs.shape

        h_out = (h_in - self.pool_size) // self.stride + 1
        w_out = (w_in - self.pool_size) // self.stride + 1

        self.output = np.zeros((batch_size, n_channels, h_out, w_out))
        self.max_indices = np.zeros((batch_size, n_channels, h_out, w_out, 2), dtype=int)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                pool_region = inputs[:, :, h_start:h_end, w_start:w_end]

                for b in range(batch_size):
                    for c in range(n_channels):
                        max_val = np.max(pool_region[b, c])
                        self.output[b, c, i, j] = max_val

                        max_idx = np.unravel_index(
                            np.argmax(pool_region[b, c]),
                            pool_region[b, c].shape
                        )
                        self.max_indices[b, c, i, j] = [max_idx[0], max_idx[1]]

    def backward(self, dvalues):
        batch_size, n_channels, h_out, w_out = dvalues.shape
        self.dinputs = np.zeros_like(self.inputs)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride

                for b in range(batch_size):
                    for c in range(n_channels):
                        max_h, max_w = self.max_indices[b, c, i, j]

                        self.dinputs[b, c, h_start + max_h, w_start + max_w] += dvalues[b, c, i, j]


class Flatten:
    def forward(self, inputs):
        self.inputs = inputs
        self.input_shape = inputs.shape
        batch_size = inputs.shape[0]
        self.output = inputs.reshape(batch_size, -1)

    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.input_shape)


class Dense:
    def __init__(self, n_inputs, n_neurons):
        # He initialization
        self.W = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.b = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.W) + self.b

    def backward(self, dvalues):
        self.dW = np.dot(self.inputs.T, dvalues)
        self.db = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.W.T)


class Dropout:
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if training:
            self.mask = np.random.binomial(1, 1-self.rate, size=inputs.shape) / (1-self.rate)
            self.output = inputs * self.mask
        else:
            self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues * self.mask


class BatchNorm:
    def __init__(self, n_features, momentum=0.9, epsilon=1e-5):
        self.n_features = n_features
        self.momentum = momentum
        self.epsilon = epsilon

        self.gamma = np.ones((1, n_features))
        self.beta = np.zeros((1, n_features))

        self.running_mean = np.zeros((1, n_features))
        self.running_var = np.ones((1, n_features))

    def forward(self, inputs, training=True):
        self.inputs = inputs

        if training:
            self.mean = np.mean(inputs, axis=0, keepdims=True)
            self.var = np.var(inputs, axis=0, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            self.normalized = (inputs - self.mean) / np.sqrt(self.var + self.epsilon)
        else:
            self.normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        self.output = self.gamma * self.normalized + self.beta

    def backward(self, dvalues):
        batch_size = dvalues.shape[0]

        self.dgamma = np.sum(dvalues * self.normalized, axis=0, keepdims=True)
        self.dbeta = np.sum(dvalues, axis=0, keepdims=True)

        dnormalized = dvalues * self.gamma
        dvar = np.sum(dnormalized * (self.inputs - self.mean) * -0.5 * (self.var + self.epsilon)**-1.5, axis=0, keepdims=True)
        dmean = np.sum(dnormalized * -1 / np.sqrt(self.var + self.epsilon), axis=0, keepdims=True) + dvar * np.sum(-2 * (self.inputs - self.mean), axis=0, keepdims=True) / batch_size

        self.dinputs = dnormalized / np.sqrt(self.var + self.epsilon) + dvar * 2 * (self.inputs - self.mean) / batch_size + dmean / batch_size
