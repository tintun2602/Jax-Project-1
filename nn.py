import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import grad
from bathtub import bathtub  


class NeuralNetworkPID:
    def __init__(self, nn_params):
        self.nn_params = nn_params
        self.I = 0.0
        self.last_error = 0.0
    
    def reset(self):
        self.I = 0.0
        self.last_error = 0.0

    def update(self, error, integral, derivative):
        input_vector = jnp.array([[error, integral, derivative]])  # Shape (1, 3)
        print(f'Input vector shape before predict call: {input_vector.shape}')
        control_signal = predict(self.nn_params, input_vector)
        return control_signal


# Define activation function
def sigmoid(x):
    """ Sigmoid activation function """
    return 1 / (1 + jnp.exp(-x))

def predict(nn_params, features):
    """Forward pass through the neural network to get the output"""
    activations = features
    for weights, biases in nn_params:
        print(f"Activations shape: {activations.shape}, Weights shape: {weights.shape}, Biases shape: {biases.shape}")
        activations = sigmoid(jnp.dot(activations, weights) + biases)
        # Activations equals the output of the net´s output
    return activations


def jaxnet_loss(params, features,targets):
    """Loss function for the neural network"""
    batched_predict = jax.vmap(predict, in_axes=(None, 0))
    predictions = batched_predict(params, features) 
    return jnp.mean(jnp.square(targets - predictions), axis=0)

# Define function to run simulation
def run_simulation(nn_pid_controller, setpoint, time_steps, disturbance_generator):
    plant = bathtub(A=10, C=0.1, H_0=setpoint)
    integral = 0.0
    last_error = 0.0
    error_history = []

    for step in range(time_steps):
        current_output = plant.get_error()
        error = setpoint - current_output
        integral += error  # Update integral
        derivative = error - last_error  # Compute derivative
        last_error = error  # Update last error for next derivative calculation
        control_signal = nn_pid_controller.update(error, integral, derivative)
        noise = disturbance_generator()
        plant.update(U=control_signal, D=noise)

        error_history.append(error)
    
    mse = jnp.mean(jnp.array(error_history) ** 2)
    return mse

# Define function to generate disturbances
def disturbance_generator(x1=-0.01, x2=0.01):
    return np.random.uniform(x1, x2)

def gen_jaxnet_params(layers=[5,10, 5]):
    """ Generate initial parameters for the neural network """
    params = []
    for input_size, output_size in zip(layers[:-1], layers[1:]):
        weights = np.random.uniform(-0.1, 0.1, (input_size, output_size))
        biases = np.random.uniform(-0.1, 0.1, (output_size,))
        params.append((weights, biases))
    return params

def jaxnet_train_one_epoch(nn_params, features, targets, lrate=0.1):
    """ Train the neural network for one epoch """
    mse, gradients  = jax.value_and_grad(jaxnet_loss)(nn_params, features, targets)
    return [(w - lrate * dw, b - lrate * db) for (w, b), (dw, db) in zip(nn_params, gradients)], mse

def jaxnet_train(nn_params, setpoint, epochs, time_steps, learning_rate):
    """ Train the neural network over multiple epochs """
    mse_history = []
    for epoch in range(epochs):
        nn_params, mse = jaxnet_train_one_epoch(nn_params, setpoint, time_steps, learning_rate)
        mse_history.append(mse)
        print(f'Epoch {epoch}: MSE = {mse}')
    return nn_params, mse_history

# Main program
def run_program():
    setpoint = 1.0  # The desired value for the plant output
    epochs = 100  # Number of training epochs
    time_steps = 1000  # Number of timesteps per epoch
    learning_rate = 0.1  # Learning rate for parameter updates

    layer_sizes = [5,10,5]  # Define the structure of the neural network
    nn_params = gen_jaxnet_params(layer_sizes)  # Generate initial NN parameters

    nn_params, mse_history = jaxnet_train(nn_params, setpoint, epochs, time_steps, learning_rate)

    # # Visualization of the training process
    # plt.figure(figsize=(10, 5))
    # plt.plot(mse_history, label='MSE', color='blue')
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean Squared Error (MSE)')
    # plt.title('MSE Over Training Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__ == '__main__':
    run_program()
