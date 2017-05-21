import numpy
import feedforward as aan

# number of the input, hidden, output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

n = aan.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_dataset/mnist_train_100.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray((all_values[1:])) / (255.0*0.99))+0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass



