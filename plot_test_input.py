import numpy
import matplotlib.pyplot

data_file = open("mnist_dataset/mnist_train_100.csv", "r")
data_list = data_file.readlines()
data_file.close()

print("Array dimensions [{}".format(len(data_list))+"]")

all_values = data_list[9].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')

matplotlib.pyplot.show()
