
import jax
import jax.numpy as jnp
from Controller import Controller

"""
AI Driven PID Controller 

"""

class NeuralNetController(Controller):
    def __init__(self, neural_network):
        super().__init__()
        self.neural_network = neural_network

    def update(self, setpoint, current_value):
        """
        
        Update the controller using the neural network

        :param setpoint: The desired target value
        :param current_value: The current process variable (measured value)
        :return: Control signal output
        
        """

        error = setpoint - current_value
        input_features = jnp.array([error])

        control_signal = self.neural_network.predict(input_features)
        return control_signal
    
    def compute_control_signal(self, error, current_time):
        return super().compute_control_signal(error, current_time)

