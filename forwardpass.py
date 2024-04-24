import numpy as np

# Define the network architecture
input_size = 2
hidden_size = 2
output_size = 2

# Initialize weights and biases
weights_input_hidden = np.array([[0.15,0.25], [0.20, 0.30]])
bias_hidden = np.array([0.35,0.60])
weights_hidden_output = np.array([[0.40,0.50], [0.45,0.55]])
bias_output = np.array([0.1, 0.99])


# Define the input
input_data = np.array([0.05, 0.10])

# Perform the forward pass
hidden_layer_input = np.dot(input_data, weights_input_hidden) + bias_hidden[0]
print(np.dot(input_data, weights_input_hidden))
print(hidden_layer_input)
hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))
print(hidden_layer_output)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_hidden[1]
output = 1 / (1 + np.exp(-output_layer_input))
# output = output + bias_output
# Print the output
print("Output:", output)