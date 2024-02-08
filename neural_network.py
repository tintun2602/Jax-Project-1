import numpy as np
import jax
import jax.numpy as jnp

from utils import disturbance_generator, PIDErrorCalculator

pid_error_calculator = PIDErrorCalculator()


class neural_network:
    def __init__(self, layers=[3, 10, 1]):
        self.layers = layers

    def gen_jaxnet_params(self):
        params = []
        for sender, receiver in zip(self.layers[:-1], self.layers[1:]):
            weights = np.random.uniform(-0.1, 0.1, (sender, receiver))
            biases = np.random.uniform(-0.1, 0.1, (receiver,))
            params.append([weights, biases])
        return params

    def sigmoid(self, x):
        return 1 / (1 + jnp.exp(-x))

    def relu(self, x):
        return jnp.maximum(0, x)

    def tanh(self, x):
        return jnp.tanh(x)

    def predict(self, params, features):
        activations = features
        for weights, biases in params:
            activations = self.tanh(jnp.dot(activations, weights) + biases)
        return activations

    def jaxnet_loss(self, params, features):
        predictions = self.predict(params, features)
        return jnp.mean(jnp.square(predictions))

    def jaxnet_train_one_epoch(self, params, features, lrate=0.1):
        mse, gradients = jax.value_and_grad(self.jaxnet_loss)(params, features)
        new_params = [(w - lrate * dw, b - lrate * db) for (w, b), (dw, db) in zip(params, gradients)]
        return new_params, mse

    def train_neural_network(self, plant, epochs=100, lrate=0.1):
        print(plant)
        params = self.gen_jaxnet_params()  # PID features to control action
        mse_history = []  # To store MSE for plotting

        for epoch in range(epochs):
            # plant = plant.reset_state()
            features = jnp.array(pid_error_calculator.calculate_errors(plant.get_error())).reshape(1, -1)
            params, mse = self.jaxnet_train_one_epoch(params, features, lrate)
            mse_history.append(mse)
            print(f'Epoch {epoch}, MSE: {mse}')

        return params, mse_history

    def control_loop(self, plant, trained_params, setpoint, time_steps, dt):
        integral_error = 0.0
        prev_error = None  # Will be used to calculate the derivative of error
        error_history = []  # Optional: track error over time for visualization
        control_signal_history = []  # Optional: track control signals over time
        water_level_history = []  # Optional: track water level over time

        for t in range(time_steps):
            current_level = plant.get_state()
            error = setpoint - current_level
            integral_error += error * dt  # Update integral of error
            derivative_error = (
                                       error - prev_error) / dt if prev_error is not None else 0.0  # Calculate derivative of error
            prev_error = error  # Update previous error for next iteration

            # Use the neural network to predict the control action
            control_action = self.predict(trained_params, np.array([[error, integral_error, derivative_error]]))
            control_signal = control_action.item()  # Assuming the NN output is a single value

            D = disturbance_generator()  # Generate a random disturbance
            # Apply the control signal to the plant
            # plant.update_state(control_signal, D, dt)  # Assuming no external disturbance (D=0)
            plant.update_state(control_signal, D)  # dt is always 1
            # Optional: Track performance over time
            error_history.append(error)
            control_signal_history.append(control_signal)
            water_level_history.append(current_level)

        print(f'Final water level: {plant.get_state()}')
