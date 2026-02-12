import numpy as np


class Convolutional:
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_height = kernel_size
            self.kernel_width = kernel_size
        else:
            self.kernel_height, self.kernel_width = kernel_size

        self.padding = padding
        self.stride = stride

        self.weights = np.random.randn(out_channels, in_channels, self.kernel_height, self.kernel_width) \
                       * np.sqrt(2 / (in_channels * self.kernel_height * self.kernel_width))
        self.biases = np.zeros(out_channels)
        
    def forward(self, inputs):
        self.inputs = inputs
        batch_size, in_channels, img_height, img_width = inputs.shape

        if self.padding > 0:
            inputs = np.pad(
                inputs,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
                constant_values=0
            )
        
        output_height = (img_height + 2 * self.padding - self.kernel_height) // self.stride + 1
        output_width = (img_width + 2 * self.padding - self.kernel_width) // self.stride + 1

        output = np.zeros((batch_size, self.out_channels, output_height, output_width))

        for batch_idx in range(batch_size):
            for out_ch in range(self.out_channels):
                for out_row in range(output_height):
                    for out_col in range(output_width):
                        h_start = out_row * self.stride
                        h_end = h_start + self.kernel_height
                        w_start = out_col * self.stride
                        w_end = w_start + self.kernel_width
                        
                        region = inputs[batch_idx, :, h_start : h_end, w_start : w_end]
                        output[batch_idx, out_ch, out_row, out_col] = np.sum(region * self.weights[out_ch]) + self.biases[out_ch]
        
        self.output = output

    def backward(self, dvalues):
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.sum(dvalues, axis=(0, 2, 3))
        self.dinputs = np.zeros_like(self.inputs)

        if self.padding > 0:
            padded_inputs = np.pad(
                self.inputs, 
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
                constant_values=0
            )
        else:
            padded_inputs = self.inputs

        padded_dinputs = np.zeros_like(padded_inputs)
        flipped_weights = np.flip(self.weights, axis=(2, 3))

        batch_size, in_channels, img_height, img_width = self.inputs.shape
        _, out_channels, out_height, out_width = dvalues.shape

        for batch_idx in range(batch_size):
            for out_ch in range(out_channels):
                for out_row in range(out_height):
                    for out_col in range(out_width):
                        h_start = out_row * self.stride
                        h_end = h_start + self.kernel_height
                        w_start = out_col * self.stride
                        w_end = w_start + self.kernel_width

                        region = padded_inputs[batch_idx, :, h_start:h_end, w_start:w_end]

                        self.dweights[out_ch] += region * dvalues[batch_idx, out_ch, out_row, out_col]

                        padded_dinputs[batch_idx, :, h_start:h_end, w_start:w_end] += (
                            flipped_weights[out_ch] * dvalues[batch_idx, out_ch, out_row, out_col]
                        )
        
        if self.padding > 0:
            self.dinputs = padded_dinputs[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            self.dinputs = padded_dinputs


class MaxPool:
    def __init__(self, kernel_size, stride=None):
        if isinstance(kernel_size, int):
            self.kernel_height = kernel_size
            self.kernel_width = kernel_size
        else:
            self.kernel_height, self.kernel_width = kernel_size

        if stride == None:
            self.stride = self.kernel_height
        else:
            self.stride = stride

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, in_channels, img_height, img_width = inputs.shape

        output_height = (img_height - self.kernel_height) // self.stride + 1
        output_width = (img_width - self.kernel_width) // self.stride + 1

        output = np.zeros((batch_size, in_channels, output_height, output_width))
        # track indices with max values for backprop
        self.max_mask = np.zeros_like(inputs)

        for batch_idx in range(batch_size):
            for out_ch in range(in_channels):
                for out_row in range(output_height):
                    for out_col in range(output_width):
                        h_start = out_row * self.stride
                        h_end = out_row * self.stride + self.kernel_height
                        w_start = out_col * self.stride
                        w_end = out_col * self.stride + self.kernel_width
                        
                        region = inputs[batch_idx, out_ch, h_start : h_end, w_start : w_end]
                        max_val = np.max(region)
                        output[batch_idx, out_ch, out_row, out_col] = max_val

                        mask = (region == max_val)
                        self.max_mask[batch_idx, out_ch, h_start : h_end, w_start : w_end] += mask
        
        self.output = output
    
    def backward(self, dvalues):
        self.dinputs = self.max_mask
        batch_size, out_channels, output_height, output_width = dvalues.shape
        
        for batch_idx in range(batch_size):
            for out_ch in range(out_channels):
                for out_row in range(output_height):
                    for out_col in range(output_width):
                        h_start = out_row * self.stride
                        h_end = out_row * self.stride + self.kernel_height
                        w_start = out_col * self.stride
                        w_end = out_col * self.stride + self.kernel_width

                        max_val = dvalues[batch_idx, out_ch, out_row, out_col]
                        n_max_vals = np.sum(self.dinputs[batch_idx, out_ch, h_start : h_end, w_start : w_end])

                        self.dinputs[batch_idx, out_ch, h_start : h_end, w_start : w_end] *= (max_val / n_max_vals)


class Flatten:
    def forward(self, inputs):
        self.input_shape = inputs.shape
        self.output = inputs.reshape(inputs.shape[0], -1)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.input_shape)