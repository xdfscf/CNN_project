import Base
import math
import numpy as np
class Sigmoid(Base.BaseLayer):

    def __init__(self):
        super(Sigmoid, self).__init__()

    def set_input(self, input):
        assert isinstance(input, np.ndarray) and input.ndim == 4, "input_layer should be a layer or 4d numpy array"
        self.input = input
        self.input_size = self.input.shape

    def sigmoid(self):
        e_x=np.power(math.e, -self.input)
        self.output=1/(1+e_x)

    def forward(self):
        self.sigmoid()

    def backward(self):
        assert self.loss is not None, "the loss passed from the following layer shouldn't be None "
        assert self.loss.shape == self.input.shape, "the shape of the loss array should equal to input"

        loss=self.output*(1-self.output)*self.loss

        return loss


