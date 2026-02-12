import numpy as np


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(inputs, 0)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_in = np.exp(self.inputs - np.max(self.inputs, axis=-1, keepdims=True))
        out = exp_in / np.sum(exp_in, axis=-1, keepdims=True)
        self.output = np.clip(out, 1e-7, 1 - 1e-7)

    def backward(self, dvalues):
        diff = self.output * (1 - self.output)
        self.dinputs = dvalues * diff


class Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.clip(1 / (1 + np.exp(-inputs)), 1e-7, 1 - 1e-7)

    def backward(self, dvalues):
        diff = self.output * (1 - self.output)
        self.dinputs = dvalues * diff


class Tanh:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)

    def backward(self, dvalues):
        diff = 1 - self.output**2
        self.dinputs = dvalues * diff
