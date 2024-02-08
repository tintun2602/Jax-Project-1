
import numpy as np

def disturbance_generator(x1=-0.01, x2=0.01):
    return np.random.uniform(x1, x2)

class PIDErrorCalculator:
    def __init__(self):
        self.integral_error = 0
        self.previous_error = 0

    def reset(self):
        self.integral_error = 0
        self.previous_error = 0

    def calculate_errors(self, error):
        self.integral_error += error
        derivative_error = error - self.previous_error
        self.previous_error = error

        return [error, self.integral_error, derivative_error]
