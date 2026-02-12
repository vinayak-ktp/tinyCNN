import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.ConvLayers import Convolutional, MaxPool, Flatten
from utils.Activations import Sigmoid
from utils.Layers import Dense
from utils.Losses import CategoricalCrossEntropy
# from utils.Optimizers import SGD


class LeNet: 
    def __init__(self):
        # Convolutional Layers
        self.conv1 = Convolutional(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2)
        self.activation1 = Sigmoid()
        self.maxpool1 = MaxPool(kernel_size=(2, 2))
        self.conv2 = Convolutional(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.activation2 = Sigmoid()
        self.maxpool2 = MaxPool(kernel_size=(2, 2))
        
        # FeedForward Layers
        self.flatten = Flatten()
        self.dense1 = Dense(n_inputs=400, n_neurons=120)
        self.activation3 = Sigmoid()
        self.dense2 = Dense(n_inputs=120, n_neurons=84)
        self.activation4 = Sigmoid()
        self.dense3 = Dense(n_inputs=84, n_neurons=10)

        # Loss Layer
        self.loss_activation = CategoricalCrossEntropy()
    
    def forward(self, X, y, type='train'):
        # Convolutional Layers (inputs.shape = (batch_size, in_channels, img_height, img_width))
        self.conv1.forward(X)
        self.activation1.forward(self.conv1.output)
        self.maxpool1.forward(self.activation1.output)
        self.conv2.forward(self.maxpool1.output)
        self.activation2.forward(self.conv2.output)
        self.maxpool2.forward(self.activation2.output)
        self.flatten.forward(self.maxpool2.output)

        # FeedForward Layers (inputs.shape = (batch_size, n_inputs))
        self.dense1.forward(self.flatten.output)
        self.activation3.forward(self.dense1.output)
        self.dense2.forward(self.activation3.output)
        self.activation4.forward(self.dense2.output)
        self.dense3.forward(self.activation4.output)

        # Loss Layer
        self.loss_activation.forward(self.dense3.output, y, type=type)

        return self.loss_activation.output

    def backward(self, y_pred, y_true):
        # FeedForward Layers
        self.loss_activation.backward(y_pred, y_true)
        self.dense3.backward(self.loss_activation.dinputs)
        self.activation4.backward(self.dense3.dinputs)
        self.dense2.backward(self.activation4.dinputs)
        self.activation3.backward(self.dense2.dinputs)
        self.dense1.backward(self.activation3.dinputs)
        self.flatten.backward(self.dense1.dinputs)

        # Convolutional Layers
        self.maxpool2.backward(self.flatten.dinputs)
        self.activation2.backward(self.maxpool2.dinputs)
        self.conv2.backward(self.activation2.dinputs)
        self.maxpool1.backward(self.conv2.dinputs)
        self.activation1.backward(self.maxpool1.dinputs)
        self.conv1.backward(self.activation1.dinputs)