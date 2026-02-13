from nn.layers import Conv2DOptimized


def optimize_conv_layers(model):
    from nn.layers import Conv2D

    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv2D):
            optimized = Conv2DOptimized(
                n_filters=layer.n_filters,
                kernel_size=layer.kernel_size,
                n_channels=layer.n_channels,
                stride=layer.stride,
                padding=layer.padding
            )
            optimized.W = layer.W.copy()
            optimized.b = layer.b.copy()
            model.layers[i] = optimized

            if hasattr(model, f'conv{i//3 + 1}'):
                setattr(model, f'conv{i//3 + 1}', optimized)

    return model
