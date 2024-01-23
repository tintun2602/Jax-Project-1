import jax.numpy as np
from jax import grad, jit

"""



A PID Controller works by controlling an output to bring a process value to a desired set point. 
It does this by calculating an error values as the difference between a desired setpoint and a measured process variable. 

The controller then applies a correction based on proportional integral, and derivative terms.
The controller attempts to minimize the error over time by adjusting a control variable to a new value determined by a weighted sum of control terms 
"""

class PIDController: 


    """
    
    
    """

    def __init__(self, Kp: float, Ki: float, Kd: float):
        
        """
        
        Parameters:

        Kp : Proportional gain.
        Ki : Integration gain.
        Kd : Derivative gain.
        
        """
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previos_error = 0
        self.integral = 0


    def update(self, setpoint, pv, dt):
        """
        
        dt: Delta Time represents the time interval between the current and previous control action
        
        """
        error = setpoint - pv
        self.integral += error * dt
        derivative = (error - self.previos_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previos_error = error
        return output
    
    
    @jit # JIT decorator - This will compile 'compute_gradient'
    def compute_gradient(self, setpoint, pv, dt):
        return grad(self.update)(setpoint, pv, dt)
    





