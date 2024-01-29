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
        ki = []
        kd = []
        kp = []

        for epoch in range(epochs):
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
            mse_history.append(mse)

        self.visualizePlot(ki, kd, kp, mse_history)

    def visualizePlot(self, ki, kd, kp, mse_history):
        mse_np = jnp.array(mse_history)

        # Visualization
        plt.figure(figsize=(18, 6))

        # Plotting mean squared error
        plt.subplot(1, 3, 1)
        plt.plot(mse_np)
        plt.xlabel('Time Steps')
        plt.axvline(x=10, color='red', linestyle='--')  # Mark x=10
        plt.ylabel('Mean Squared Error')
        plt.title('Mean Squared Error Over Time')
        plt.ylim(0, 1)
        plt.xlim(0, 100)

    # Plotting the water level in the bathtub

        plt.subplot(1, 3, 2)
        # plt.plot(self.plant.state_history) 
        plt.xlabel('Time Steps')
        plt.ylabel('Water Level')
        plt.title('Water Level Over Time')
        plt.ylim(0, 100)
        plt.xlim(0, 100)


        plt.subplot(1, 3, 3)
        plt.plot(ki, label='ki')
        plt.plot(kd, label='kd')
        plt.plot(kp, label='kp')
        plt.xlabel('Time Steps')
        plt.axvline(x=10, color='red', linestyle='--')  # Mark x=10
        plt.ylabel('Mean Squared Error')
        plt.title('Mean Squared Error Over Time')
        plt.ylim(0, 1)
        plt.xlim(0, 100)

        plt.tight_layout()
        plt.show()


    
        
