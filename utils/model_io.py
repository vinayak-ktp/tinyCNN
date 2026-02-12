import pickle

import numpy as np


def save_model(model, filepath):
    model_data = {
        'class': model.__class__.__name__,
        'input_shape': model.input_shape,
        'n_classes': model.n_classes,
        'layers': []
    }

    for layer in model.layers:
        layer_data = {'type': layer.__class__.__name__}

        if hasattr(layer, 'W'):
            layer_data['W'] = layer.W
        if hasattr(layer, 'b'):
            layer_data['b'] = layer.b
        if hasattr(layer, 'gamma'):
            layer_data['gamma'] = layer.gamma
        if hasattr(layer, 'beta'):
            layer_data['beta'] = layer.beta
        if hasattr(layer, 'running_mean'):
            layer_data['running_mean'] = layer.running_mean
        if hasattr(layer, 'running_var'):
            layer_data['running_var'] = layer.running_var

        model_data['layers'].append(layer_data)

    np.savez(filepath, **model_data)


def load_model(filepath, model_class):
    data = np.load(filepath, allow_pickle=True)

    model = model_class(
        input_shape=tuple(data['input_shape']),
        n_classes=int(data['n_classes'])
    )

    layers_data = data['layers']

    for layer, layer_data in zip(model.layers, layers_data):
        layer_dict = layer_data.item()

        if 'W' in layer_dict:
            layer.W = layer_dict['W']
        if 'b' in layer_dict:
            layer.b = layer_dict['b']
        if 'gamma' in layer_dict:
            layer.gamma = layer_dict['gamma']
        if 'beta' in layer_dict:
            layer.beta = layer_dict['beta']
        if 'running_mean' in layer_dict:
            layer.running_mean = layer_dict['running_mean']
        if 'running_var' in layer_dict:
            layer.running_var = layer_dict['running_var']

    return model
