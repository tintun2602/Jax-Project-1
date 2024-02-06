# Since the implementation is already provided, let's execute the provided code directly.

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import random, value_and_grad

class BathtubPlant:
    def __init__(self, initial_level, area, drain_area, g=9.8):
        self.initial_level = jnp.maximum(initial_level, 0)
        self.A = area  # Area of the bathtub
        self.C = drain_area  # Area of the drain
        self.g = g  # Acceleration due to gravity
        self.reset()  # Use reset to initialize water level, V, and Q based on initial_level

    def get_state(self):
        return self.water_level

    def update_state(self, control_signal, D):
        V = jnp.sqrt(2 * self.g * self.water_level)
        Q = V * self.C
        delta_B = control_signal + D - Q
        
        assert self.A != 0, "Area is zero, leading to division by zero."
        delta_H = delta_B / self.A
        
        new_water_level = self.water_level + delta_H
        if new_water_level < 0:
            new_water_level = 0  # Prevent negative water level
        
        self.water_level = new_water_level
        return self.water_level

    def reset(self):
        self.water_level = self.initial_level
        self.V = jnp.sqrt(2 * self.g * self.water_level) 
        self.Q = self.V * self.C

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def predict(params, x):
    activations = x
    for w, b in params[:-1]:
        activations = sigmoid(jnp.dot(activations, w) + b)
    final_w, final_b = params[-1]
    return jnp.dot(activations, final_w) + final_b

def mse_loss(params, inputs, targets):
    preds = predict(params, inputs)
    return jnp.mean((preds - targets) ** 2)

def update_params(params, gradients, lrate):
    return [(w - lrate * dw, b - lrate * db) for (w, b), (dw, db) in zip(params, gradients)]

def jaxnet_train_one_epoch(params, features, targets, lrate=0.1):
    mse, gradients = value_and_grad(mse_loss)(params, features, targets)
    params = update_params(params, gradients, lrate)
    return params, mse

def gen_jaxnet_params(layer_sizes):
    rng = np.random.default_rng()
    return [(rng.standard_normal((i, j)), rng.standard_normal((j,)))
            for i, j in zip(layer_sizes[:-1], layer_sizes[1:])]

def simulate_and_train_bathtub(params, key, epochs=100, steps_per_epoch=100, target_level=1.0):
    bathtub = BathtubPlant(initial_level=1.0, area=10.0, drain_area=0.1)
    mse_history = []  # Track MSE for each epoch

    for epoch in range(epochs):
        bathtub.reset()
        error_history = []  # Reset error history each epoch
        control_signal_history = []  # Reset control signal history each epoch

        for step in range(steps_per_epoch):
            key, subkey = random.split(key)
            current_level = bathtub.get_state()
            error = target_level - current_level
            error_array = jnp.array([error]).reshape(1, -1)

            control_signal = predict(params, error_array)
            control_signal_history.append(control_signal)

            D = random.uniform(subkey, shape=(), minval=-0.01, maxval=0.01)
            bathtub.update_state(control_signal[0], D)

            error_history.append([error])

        # Convert list of lists (or list of JAX arrays) to a JAX array
        features = jnp.array([jnp.array(e).reshape(1,) for e in error_history]).reshape(-1, 1)

        targets = jnp.array(control_signal_history).reshape(-1, 1)
        params, mse = jaxnet_train_one_epoch(params, features, targets, lrate=0.2)
        mse_history.append(mse)

    return params, mse_history

# Initialize the parameters and key
key = random.PRNGKey(0)
layer_sizes = [1, 40, 1]  # Neural network architecture
params = gen_jaxnet_params(layer_sizes)

# Run the simulation and training
params, mse_history = simulate_and_train_bathtub(params, key)

# Corrected plotting section in the simulation and training function
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(mse_history) + 1), mse_history, label='MSE', color='black')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Learning Progress Over Epochs')
plt.legend()
plt.show()