import numpy as np


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
