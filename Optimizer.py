import numpy as np

import CNN


class CustomOptimizer:
    def __init__(self, learning_rate, layers, loss_function):
        self.learning_rate = learning_rate
        self.momentums = {}  # Store momentums for each layer
        self.layers=layers
        for layer in self.layers:
            if layer not in self.momentums and isinstance(layer, CNN.CNN):
                self.momentums[layer] = np.zeros_like(layer.kernel)
        self.loss_func=loss_function

    def update(self, input, target):

        for layer in self.layers:
            layer.set_input(input)
            layer.forward()
            input=layer.get_output()

        output=self.layers[-1].get_output()
        loss=self.loss_func(output,target)
        loss=np.asarray(loss)
        loss=np.reshape(loss, output.shape)
        print(loss)
        for layer in self.layers[1:][::-1]:
            layer.set_loss(loss)
            if isinstance(layer, CNN.CNN):
                gradient = layer.gradient()
                self.momentums[layer] = 0.9 * self.momentums[layer] + gradient
                update_term = self.learning_rate * self.momentums[layer]
                layer.kernel += update_term
            loss=layer.backward()
