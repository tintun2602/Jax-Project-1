import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import random


# 

def disturbance_generator(x1=-1, x2=1):
    """

    :param x1: min value of disturbance
    :param x2: max value of disturbance
    :return: random value between min and max value
    """ 

    return np.random.uniform(x1, x2)


# TODO: Implement compute_gradients function in the ControlSystem class

def compute_gradients(self, error_history, ki_history, kd_history, kp_history):
    num_timesteps = len(error_history)
    dMSE_dki = 2 * np.sum(np.array(error_history) * np.array(ki_history)) / num_timesteps
    dMSE_dkd = 2 * np.sum(np.array(error_history) * np.array(kd_history)) / num_timesteps
    dMSE_dkp = 2 * np.sum(np.array(error_history) * np.array(kp_history)) / num_timesteps
    return {'ki': dMSE_dki, 'kd': dMSE_dkd, 'kp': dMSE_dkp}


def update_controller_params(self, gradients, learning_rate=0.1):
    self.controller.ki += learning_rate * gradients['ki']
    self.controller.kd += learning_rate * gradients['kd']
    self.controller.kp += learning_rate * gradients['kp']


class ControlSystem:
    def __init__(self, controller, plant):
        self.controller = controller
        self.plant = plant

    def run_control_loop(self, setpoint, epochs, timesteps):
        mse_history = []

        # initializing water height history
        water_height_history = []
        ki = []
        kd = []
        kp = []

        for epoch in range(epochs):
            # resetting the plant to its initial state.
            self.plant.reset()
            
            # initializing error history
            error_history = []

            for timestep in range(timesteps):
                if timestep == 0:
                    continue
                disturbance = disturbance_generator()

                # computing error and control signal
                error, control_signal = self.controller.update(setpoint, self.plant.state, timestep)
                self.plant.update(control_signal, disturbance)
                error_history.append(error)

                # storing the parameters
                ki.append(self.controller.ki)
                kd.append(self.controller.kd)
                kp.append(self.controller.kp)

            mse = np.mean(np.square(np.array(error_history)))

            # adding data points to the history
            mse_history.append(mse)
            water_height_history.append(self.plant.state)

        self.visualizePlot(ki, kd, kp, mse_history)

    def visualizePlot(self, kp, ki, kd, mse_history):
        mse_np = np.array(mse_history)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(mse_np)
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('MSE')
        axs[0].set_title('Learning Progression')
        axs[0].grid(True)
        axs[1].plot(kp, label='Kp')
        axs[1].plot(ki, label='Ki')
        axs[1].plot(kd, label='Kd')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Value')
        axs[1].set_title('Control Parameters')
        axs[1].legend()
        axs[1].grid(True)
        plt.tight_layout()
        plt.show()


