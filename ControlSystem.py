import numpy as np
import jax
import jax.numpy as jnp


class ControlSystem: 
    def __init__(self, controller, plant):
        self.controller = controller
        self.plant = plant 

    def run_control_loop(self, setpoint,  disturbance_generator, epochs, timesteps):
        for epoch in range(epochs):
            error_history = []
            for timestep in range(timesteps):
                disturbance = disturbance_generator()
                control_signal = self.controller.compute_control_signal(self.plant.state)
                self.plant.update(control_signal, disturbance)
                error = setpoint - self.plant.state
                error_history.append(error)
                self.controller.update(setpoint, self.plant.state, timestep)

            mse = np.mean(np.square(np.array(error_history)))