from Matrix_class import *

class NN():
    def __init__(self):

        # Shape [2 2 1]
        self.input = Matrix(2, 1)
        self.input[0,0] = 1
        self.input[1,0] = 0

        self.W_1 = Matrix(2, 2)
        self.W_1[0, 0] = -5.8460097
        self.W_1[0, 1] = -5.6625414
        self.W_1[1, 0] = -3.8732808
        self.W_1[1, 1] = -3.8040390

        self.b_1 = Matrix(2, 1)
        self.b_1[0, 0] = 2.1355941
        self.b_1[0, 1] = 5.6442757

        self.W_2 = Matrix(1, 2)
        self.W_2[0, 0] = -7.8511896
        self.W_2[0, 1] = 7.540539

        self.b_2 = Matrix(1, 1)
        self.b_2[0, 0] = -3.4388447

        self.output_ = None

    def update(self):
        output_first_layer = (self.W_1 * self.input + self.b_1).sigmoid
        output_second_layer = (self.W_2 * output_first_layer + self.b_2).sigmoid
        self.output_ = output_second_layer


network = NN()

network.update()

print(network.output_)