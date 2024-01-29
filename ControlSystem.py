import numpy as np
import jax.numpy as jnp 
import matplotlib.pyplot as plt

def disturbance_generator(x1=-1, x2=1):
    """

    :param x1: min value of disturbance
    :param x2: max value of disturbance
    :return: random value between min and max value
    """

    return np.random.uniform(x1, x2)


class ControlSystem:
    def __init__(self, controller, plant):
        self.controller = controller
        self.plant = plant

    def run_control_loop(self, setpoint, epochs, timesteps):
        mse_history = []
        for epoch in range(epochs):
            error_history = []
            for timestep in range(timesteps):
                if timestep == 0:
                    continue
                disturbance = disturbance_generator()

                # Error:
                error = setpoint - self.plant.state
                control_signal = self.controller.compute_control_signal(error, timestep)

                self.plant.update(control_signal, disturbance)
                error = setpoint - self.plant.state
                print(f"Error: {error}")
                error_history.append(error)
                self.controller.update(setpoint, self.plant.state, timestep)

            mse = np.mean(np.square(np.array(error_history)))
            mse_history.append(mse)

        self.plot(mse_history)

    def plot(self, mse_history):
        mse_np = jnp.array(mse_history)

        # Visualization
        plt.figure(figsize=(12, 6))

        # Plotting mean squared error
        plt.subplot(1, 2, 1)
        plt.plot(mse_np)
        plt.xlabel('Time Steps')
        plt.axvline(x=10, color='red', linestyle='--')  # Mark x=10
        plt.ylabel('Mean Squared Error')
        plt.title('Mean Squared Error Over Time')
        plt.ylim(0, 1)
        plt.xlim(0, 100)

        plt.tight_layout()
        plt.show()
