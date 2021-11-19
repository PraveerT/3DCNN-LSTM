# Python program to implement a
# single neuron neural network

# import all necessary libraries
from numpy import exp, array, random, dot, tanh,sin,cos


# Class to create a neural
# network with single neuron
class NeuralNetwork():

    def __init__(self,train_input,seednum):
        # Using seed to make sure it'll
        # generate same weights in every run
        random.seed(seednum)

        # 3x1 Weight matrix
        self.complex=1.j
        self.weight_matrix = 2 * random.random((len(train_input[1]), 1)) - random.random((len(train_input[1]), 1))*self.complex

    # tanh as activation function
    def tanh(self, x):
        return tanh(x)

    def xsin(self,x):
        return x-sin(x)

    # derivative of tanh function.
    # Needed to calculate the gradients.
    def xsin_derivative(self, x):
        return 1-cos(x)
    def tanh_derivative(self, x):
        return 1.0 - tanh(x) ** 2

    def sigmoid_derivative(self,x):
        return exp(-x)/((1+exp(-x))**2)

    # forward propagation
    def forward_propagation(self, inputs):

        return self.tanh(dot(inputs, self.weight_matrix))

    # training the neural network.
    def train(self, train_inputs, train_outputs,
              num_train_iterations):
        # Number of iterations we want to
        # perform for this set of input.
        for iteration in range(num_train_iterations):
            output = self.forward_propagation(train_inputs)

            # Calculate the error in the output.
            error = train_outputs - output

            # multiply the error by input and then
            # by gradient of tanh function to calculate
            # the adjustment needs to be made in weights
            adjustment = dot(train_inputs.T, error *
                             self.tanh_derivative(output))

            # Adjust the weight matrix
            self.weight_matrix += adjustment


# Driver Code
if __name__ == "__main__":

    train_inputs = array([[0.j,0.j, 0.j, 0.j], [1.j, 1.j, 1.j, 1.j], [1.j, 1.j, 1.j, 0.j], [0.j,0.j,0.j, 0.j]])
    train_outputs = array([[0, 1, 1, 0]]).T
    sn=3
    neural_network = NeuralNetwork(train_inputs,sn)

    print('Random weights at the start of training')
    print(neural_network.weight_matrix)


    neural_network.train(train_inputs, train_outputs, 100000)

    print('New weights after training')
    print(neural_network.weight_matrix)

    # Test the neural network with a new situation.
    print("Testing network on new examples ->")
    print(neural_network.forward_propagation(array([1, 1, 1, 1])))
