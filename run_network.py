# %% Imports
import mnist_loader
# import network
import network_matrix

# %% Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# %% Network with 30 neurons and two hidden layers
net = network_matrix.Network([784, 30, 10])
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

# %%
