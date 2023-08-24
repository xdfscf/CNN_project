class BaseLayer():
    def __init__(self):
        self.input = None
        self.output = None
        self.loss = None
        self.previous_layer=None
        self.gradiants=None
        self.kernel=None

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output

    def set_loss(self, loss):
        self.loss = loss

    def set_input(self, input):
        self.input=input

    def forward(self):
        pass

    def backward(self):
        pass

    def gradient(self):
        pass

