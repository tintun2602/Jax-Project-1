import jax.numpy as np
from jax import grad, jit



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
