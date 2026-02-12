import numpy as np
from nn.activations import ReLU, Sigmoid, Softmax
from nn.layers import BatchNorm, Conv2D, Dense, Dropout, Flatten, MaxPool2D


class SimpleCNN:
    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape  # (channels, height, width)
        self.n_classes = n_classes

        self.conv1 = Conv2D(n_filters=32, kernel_size=3, n_channels=input_shape[0], stride=1, padding=1)
        self.relu1 = ReLU()

        self.conv2 = Conv2D(n_filters=32, kernel_size=3, n_channels=32, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)

        self.conv3 = Conv2D(n_filters=64, kernel_size=3, n_channels=32, stride=1, padding=1)
        self.relu3 = ReLU()

        self.conv4 = Conv2D(n_filters=64, kernel_size=3, n_channels=64, stride=1, padding=1)
        self.relu4 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)

        self.conv5 = Conv2D(n_filters=128, kernel_size=3, n_channels=64, stride=1, padding=1)
        self.relu5 = ReLU()

        self.conv6 = Conv2D(n_filters=128, kernel_size=3, n_channels=128, stride=1, padding=1)
        self.relu6 = ReLU()
        self.pool3 = MaxPool2D(pool_size=2, stride=2)

        h_out = input_shape[1] // 8  # 3 pooling layers
        w_out = input_shape[2] // 8
        flattened_size = 128 * h_out * w_out

        self.flatten = Flatten()

        self.fc1 = Dense(flattened_size, 512)
        self.relu7 = ReLU()
        self.dropout1 = Dropout(rate=0.5)

        self.fc2 = Dense(512, 256)
        self.relu8 = ReLU()
        self.dropout2 = Dropout(rate=0.5)

        self.fc3 = Dense(256, n_classes)

        self.layers = [
            self.conv1, self.relu1,
            self.conv2, self.relu2, self.pool1,
            self.conv3, self.relu3,
            self.conv4, self.relu4, self.pool2,
            self.conv5, self.relu5,
            self.conv6, self.relu6, self.pool3,
            self.flatten,
            self.fc1, self.relu7, self.dropout1,
            self.fc2, self.relu8, self.dropout2,
            self.fc3
        ]

    def forward(self, inputs, training=True):
        self.inputs = inputs
        x = inputs

        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.forward(x, training=training)
            else:
                layer.forward(x)
            x = layer.output

        self.output = x
        return self.output

    def backward(self, dvalues):
        dx = dvalues

        for layer in reversed(self.layers):
            layer.backward(dx)
            dx = layer.dinputs


class TinyCNN:
    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes

        self.conv1 = Conv2D(n_filters=16, kernel_size=3, n_channels=input_shape[0], stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)

        self.conv2 = Conv2D(n_filters=32, kernel_size=3, n_channels=16, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)

        h_out = input_shape[1] // 4
        w_out = input_shape[2] // 4
        flattened_size = 32 * h_out * w_out

        self.flatten = Flatten()

        self.fc1 = Dense(flattened_size, 128)
        self.relu3 = ReLU()
        self.dropout = Dropout(rate=0.5)

        self.fc2 = Dense(128, n_classes)

        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten,
            self.fc1, self.relu3, self.dropout,
            self.fc2
        ]

    def forward(self, inputs, training=True):
        self.inputs = inputs
        x = inputs

        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.forward(x, training=training)
            else:
                layer.forward(x)
            x = layer.output

        self.output = x
        return self.output

    def backward(self, dvalues):
        dx = dvalues

        for layer in reversed(self.layers):
            layer.backward(dx)
            dx = layer.dinputs
