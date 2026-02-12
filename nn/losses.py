import numpy as np

from nn.activations import Softmax


class CategoricalCrossEntropyLoss:
    def __init__(self):
        self.activation = Softmax()

    def forward(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

        self.activation.forward(inputs)
        self.predictions = self.activation.output

        batch_size = len(targets)

        # Handle one-hot/integer targets
        if len(targets.shape) == 1:
            correct_probs = self.predictions[np.arange(batch_size), targets]
        else:
            correct_probs = np.sum(self.predictions * targets, axis=1)

        correct_probs = np.clip(correct_probs, 1e-7, 1 - 1e-7)

        return -np.mean(np.log(correct_probs))

    def backward(self):
        batch_size = len(self.targets)

        # Handle one-hot/integer targets
        if len(self.targets.shape) == 1:
            vocab_size = self.predictions.shape[1]
            targets_one_hot = np.zeros((batch_size, vocab_size))
            targets_one_hot[np.arange(batch_size), self.targets] = 1
        else:
            targets_one_hot = self.targets

        self.dinputs = (self.predictions - targets_one_hot) / batch_size


class BinaryCrossEntropyLoss:
    def forward(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

        self.predictions = np.clip(1 / (1 + np.exp(-inputs)), 1e-7, 1 - 1e-7)

        loss = -(targets * np.log(self.predictions) + (1 - targets) * np.log(1 - self.predictions))
        return np.mean(loss)

    def backward(self):
        batch_size = len(self.targets)
        self.dinputs = (self.predictions - self.targets) / batch_size
