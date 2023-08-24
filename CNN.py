import math
import Activation
import Base
import numpy as np


class MyCustomException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class CNN(Base.BaseLayer):

    def __init__(self, in_channel, out_channel, kernel, stride, pattern):
        super(CNN, self).__init__()

        self.in_channel = in_channel

        self.out_channel = out_channel

        if isinstance(kernel, tuple) and len(kernel) == 2:
            self.kernel = np.random.randn(out_channel, in_channel, kernel[0], kernel[1]) / np.sqrt(
                kernel[0] * kernel[1] * in_channel)

        elif isinstance(kernel, int):
            self.kernel = np.random.randn(out_channel, in_channel, kernel, kernel) / np.sqrt(
                kernel * kernel * in_channel)
        else:
            raise MyCustomException("kernel must be a 2d-tuple or a number")

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple) and len(stride) == 2:
            self.stride = stride
        else:
            raise MyCustomException("stride must be a 2d-tuple or a number")

        assert pattern in {"same", "valid"}, "pattern should be either same or valid"
        self.pattern = pattern


    def set_input(self, input):
        if isinstance(input, np.ndarray) and input.ndim == 4:
            self.input = input
            self.original_input_size=input.shape
        else:
            raise MyCustomException("input_layer should be a layer or 4d numpy array")

        assert self.input.shape[1] == self.in_channel, "incorrect channel dimension"

        self.input_size = self.input.shape

        if self.pattern == "same":

            output_height = int(math.floor(self.input_size[-2] / self.stride[-2]))
            output_width = int(math.floor(self.input_size[-1] / self.stride[-1]))

            if self.input_size[-2] % self.stride[-2] == 0:
                total_padding_h = max(self.kernel.shape[-2] - self.stride[-2], 0)
            else:
                total_padding_h = max(self.kernel.shape[-2] - self.input_size[-2] % self.stride[-2], 0)

            if self.input_size[-1] % self.stride[-1] == 0:
                total_padding_w = max(self.kernel.shape[-1] - self.stride[-1], 0)
            else:
                total_padding_w = max(self.kernel.shape[-1] - self.input_size[-1] % self.stride[-1], 0)

            padding_h_t = int(total_padding_h / 2)
            padding_h_b = total_padding_h - padding_h_t
            padding_w_l = int(total_padding_w / 2)
            padding_w_r = total_padding_w - padding_w_l

            self.output = np.zeros((self.input_size[0], self.out_channel, output_height, output_width))
            self.output_size = self.output.shape
            self.input = np.pad(self.input,
                                ((0, 0),
                                 (0, 0),
                                 (padding_h_t, padding_h_b),
                                 (padding_w_r, padding_w_l)
                                 )
                                )

            self.input_size = self.input.shape

        else:
            output_height = int(math.ceil((self.input_size[-2] - self.kernel.shape[-2]) / self.stride[-2]) + 1)
            output_width = int(math.ceil((self.input_size[-1] - self.kernel.shape[-1]) / self.stride[-1]) + 1)
            self.output = np.zeros((self.input_size[0], self.out_channel, output_height, output_width), dtype=np.float64)
            self.output_size = self.output.shape

    def conv(self,output_size, stride, input, kernel):
        h = output_size[2]
        w = output_size[3]
        input_size=input.shape
        input_h = 0
        input_w = 0

        conv_result=np.zeros(output_size, dtype=np.float64)
        for height in range(h):
            for width in range(w):
                for input_ind in range(input_size[0]):
                    for kernel_ind in range(kernel.shape[0]):

                        conv_result[input_ind][kernel_ind][height][width] = \
                            np.sum(
                                input[input_ind][:, input_h:input_h + kernel.shape[-2],
                                input_w:input_w + kernel.shape[-1]] * kernel[kernel_ind])
                input_w += stride[-1]

            input_w = 0
            input_h += stride[-2]
        return conv_result

    def loss_conv(self,output_size, stride, input, kernel):
        h = output_size[2]
        w = output_size[3]
        input_size=input.shape
        input_h = 0
        input_w = 0

        conv_result=np.zeros(output_size, dtype=np.float64)
        for height in range(h):
            for width in range(w):
                for input_ind in range(input_size[0]):
                    for kernel_ind in range(kernel.shape[0]):
                        conv_result[input_ind][:, height, width] += \
                                np.sum(
                                    input[input_ind][kernel_ind][input_h:input_h + kernel.shape[-2],
                                    input_w:input_w + kernel.shape[-1]] * kernel[kernel_ind], axis=(-1,-2))
                input_w += stride[-1]

            input_w = 0
            input_h += stride[-2]
        return conv_result

    def gradiant_conv(self,kernel_size, stride, input, loss):

        h = kernel_size[2]
        w = kernel_size[3]
        input_size=input.shape
        input_h = 0
        input_w = 0

        kernel_gradiant=np.zeros(kernel_size, dtype=np.float64)
        for height in range(h):
            for width in range(w):
                for input_ind in range(input_size[0]):
                    for kernel_ind in range(loss.shape[1]):
                        kernel_gradiant[kernel_ind][:, height, width] += \
                            np.sum(input[input_ind][:, input_h:input_h + loss.shape[-2],
                                    input_w:input_w + loss.shape[-1]] * loss[input_ind][kernel_ind], axis=(-1,-2))

                input_w += stride[-1]

            input_w = 0
            input_h += stride[-2]
        return kernel_gradiant

    def forward(self):
        assert self.input is not None, "the input shouldn't be none"
        self.output=self.conv(self.output_size, self.stride, self.input, self.kernel)

    def backward(self):
        assert self.loss is not None, "the loss passed from the following layer shouldn't be None "
        h=self.output_size[2]
        w=self.output_size[3]
        stride_h=self.stride[0]
        stride_w=self.stride[1]
        stride_loss_h=h+(h-1)*(stride_h-1)
        stride_loss_w=w+(w-1)*(stride_w-1)
        loss_with_stride=np.zeros((self.output_size[0],self.output_size[1],stride_loss_h,stride_loss_w), dtype=np.float64)

        loss_h = 0
        loss_w = 0

        for height in range(h):
            for width in range(w):
                for batch_ind in range(self.output_size[0]):
                    for channel_ind in range(self.output_size[1]):

                        loss_with_stride[batch_ind][channel_ind][loss_h][loss_w] = self.loss[batch_ind][channel_ind][height][width]

                loss_w+=stride_w
            loss_w = 0
            loss_h += stride_h

        h_padding=int((self.kernel.shape[2]-loss_with_stride.shape[2] -1 +self.original_input_size[2])/2)
        w_padding=int((self.kernel.shape[3]-loss_with_stride.shape[3]-1 +self.original_input_size[3])/2)
        loss_with_stride=np.pad(loss_with_stride,
                                ((0,0),
                                 (0,0),
                                 (h_padding,h_padding),
                                 (w_padding,w_padding)
                                ))
        rotated_kernel = np.flip(np.flip(self.kernel, axis=2), axis=3)

        loss = self.loss_conv(self.original_input_size, (1,1), loss_with_stride, rotated_kernel)
        '''
        if self.previous_layer is not None:
            self.previous_layer.set_loss(loss)
        '''
        return loss


    def gradient(self):
        assert self.loss is not None, "the loss passed from the following layer shouldn't be None "
        self.gradiants = self.gradiant_conv(self.kernel.shape, self.stride, self.input, self.loss)
        return self.gradiants





