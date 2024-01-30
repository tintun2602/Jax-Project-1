import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import random

from plants.BathtubPlant import BathubPlant



class ClassicPIDController():
    def __init__(self, kp, ki, kd):
        super().__init__()  # Initialize the base class
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_sum = 0
        self.previous_error = 0

    def compute_control_signal(self, error, dt):
        if dt == 0: 
            raise ValueError("Delta time cannot be zero")
        
        # Calculate the derivative of the error
        derivative_error = (error - self.previous_error) / dt
        
        # Calculate the integral of the error
        self.integral_sum += error * dt
        
        # PID formula: output = P + I + D
        control_signal = (self.kp * error) + (self.ki * self.integral_sum) + (self.kd * derivative_error)
        
        # Update the previous error for the next iteration
        self.previous_error = error
        
        return control_signal

    def update(self, setpoint, pv, dt):
        # Compute error and call compute_control_signal to get the PID output
        error = setpoint - pv
        output = self.compute_control_signal(error, dt)
        return error, output


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

def main():
    # creating an instance of the bathtub plant
    cross_sectional_area_of_bathtub = 10
    cross_sectional_area_of_bathtub_drain = 0.01
    initial_and_goal_height_of_water_in_bathtub = 100

    bathtub_plant = BathubPlant(
        cross_sectional_area_of_bathtub,
        cross_sectional_area_of_bathtub_drain,
        initial_and_goal_height_of_water_in_bathtub
    )

    # creating an instance of the classical_pid_controller
    classical_pid = ClassicPIDController(kp=1.0, ki=0.1, kd=0.01)

    controll_system = ControlSystem(classical_pid, bathtub_plant)

    epochs = 100
    timesteps = 100

    controll_system.run_control_loop(
        initial_and_goal_height_of_water_in_bathtub,
        epochs,
        timesteps
    )

if __name__ == "__main__":
    main()
