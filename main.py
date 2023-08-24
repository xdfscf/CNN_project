import numpy as np

import CNN
import Activation
import Optimizer
import Loss_function
if __name__ == "__main__":
    input = np.random.rand(1, 1, 20, 20)
    target=1.0
    in_channel = 1
    out_channel = 3
    kernel_size = 3
    stride_size = (2, 2)
    loss=np.random.rand(1,3,2,2)
    pattern = "same"
    Cnn_layer1 = CNN.CNN(in_channel, out_channel, kernel_size, stride_size, pattern)
    activation_layer1=Activation.Sigmoid()
    Cnn_layer2 = CNN.CNN(3, 20, 4, 1, "valid")
    activation_layer2 = Activation.Sigmoid()
    Cnn_layer3 = CNN.CNN(20, 1, 4, 1, "valid")
    activation_layer3 = Activation.Sigmoid()
    Cnn_layer4 = CNN.CNN(1, 1, 4, 1, "valid")
    activation_layer4 = Activation.Sigmoid()
    layers=[Cnn_layer1,activation_layer1, Cnn_layer2,activation_layer2,Cnn_layer3,activation_layer3, Cnn_layer4, activation_layer4]
    Optimizer=Optimizer.CustomOptimizer(learning_rate=0.1, layers=layers, loss_function=Loss_function.mse_loss)
    train_epoch=50
    for i in range(train_epoch):
        Optimizer.update(input, target)


