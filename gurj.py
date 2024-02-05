import jax
import jax.numpy as jnp
from jax import jit, random
import matplotlib.pyplot as plt

# Constants for the bathtub model using JAX numpy
A = 10  # Cross-sectional area of the bathtub
C = 0.1  # Cross-sectional area of the drain
g = 9.8  # Gravitational constant

class BathtubPlant:

    def __init__(self, initial_level, area, drain_area, g=9.8):
        # Ensure initial water level is non-negative
        self.water_level = initial_level
        self.A = area
        self.C = drain_area
        self.g = g
        self.V = jnp.sqrt(2 * self.g * self.water_level)
        self.Q = self.V * self.C  

        # Used for resetting plant
        self.intial_height = initial_level
    
    def get_state(self):
        return self.water_level

    def update_state(self, control_signal, D):
        self.V = jnp.sqrt(2 * self.g * self.water_level)
        self.Q = self.V * self.C  
        delta_B = control_signal + D - self.Q
        delta_H = delta_B / self.A
        self.water_level += delta_H
        return self.water_level
    
    def reset(self):
        self.water_level = self.intial_height
        self.V = jnp.sqrt(2 * self.g * self.water_level)
        self.Q = self.V * self.C


# PID Controller using JAX
class PIDController:
    def __init__(self, kp, ki, kd, set_point):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_point = set_point
        self.integral = 0
        self.prev_error = 0

    def update(self, current_value):
        """Update the PID controller using JAX."""
        # error = self.set_point - current_value
        # self.integral += error
        # derivative = error - self.prev_error
        # self.prev_error = error

        error = self.set_point - current_value
        P = self.kp * error
        self.integral += error
        derivative = (error - self.prev_error)
        self.prev_error = error

        # mse = self.mse_fun(current_value)

        return P + (self.ki * self.integral) + (self.kd * derivative)

        # return self.kp + self.ki * self.integral + self.kd * derivative
    
    def reset(self):
        self.integral = 0
        self.prev_error = 0


# Simulation parameters
num_epochs = 100
num_timesteps = 100
# initial_height = jnp.array(1.0)  # Starting water height
initial_height = 20.0  # Starting water height
# target_height = jnp.array(1.0)  # Target water height
target_height = 20.0
pid_params = {'kp': 0.1, 'ki': 5, 'kd': 3}  # PID parameters
key = random.PRNGKey(0)  # Random seed for JAX

bathtub = BathtubPlant(initial_height, A, C, g)

# Initialize PID controller
pid = PIDController(**pid_params, set_point=target_height)

# Simulation and Visualization
heights = []
errors = []

kp_values, ki_values, kd_values = [], [], []


# Define a loss function for the PID Controller
def pid_loss(kp, ki, kd, set_point, initial_height, num_timesteps, key):
    pid = PIDController(kp, ki, kd, set_point)
    bathtub = BathtubPlant(initial_height, A, C, g)
    total_error = 0.0
    
    for t in range(num_timesteps):
        key, subkey = random.split(key)
        D = random.uniform(subkey, (), minval=-0.01, maxval=0.01)  # Random disturbance/noise
        current_height = bathtub.get_state()
        U = pid.update(current_height)
        bathtub.update_state(U, D)
        error = set_point - current_height
        total_error += error**2
    
    mse = total_error / num_timesteps
    return mse


# Partially apply pid_loss to prepare it for gradient computation
def make_loss_function(set_point, initial_height, num_timesteps, key):
    def loss_fn(kp, ki, kd):
        return pid_loss(kp, ki, kd, set_point, initial_height, num_timesteps, key)
    return loss_fn
    # Update PID parameters if using adaptive control
    # if adaptive:    

 # Initialize simulation and PID parameters
num_epochs = 1000
num_timesteps = 100
initial_height = 1.0  # Starting water height
target_height = 1.0  # Target water height
pid_params = {'kp': 0.1, 'ki': 0.05, 'kd': 0.01}  # Initial PID parameters
learning_rate = 0.01
key = random.PRNGKey(0)  # Random seed for JAX

# Prepare the loss function for gradients
loss_fn = make_loss_function(target_height, initial_height, num_timesteps, key)

# Compute gradients of the loss function w.r.t. PID parameters
grad_loss_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
grad_jit = jax.jit(grad_loss_fn)

# Lists to store MSE values over epochs
mse_values = []

for epoch in range(num_epochs):
    # Compute gradients
    grads = grad_jit(pid_params['kp'], pid_params['ki'], pid_params['kd'])
    
    # Update PID parameters
    pid_params['kp'] -= learning_rate * grads[0]
    pid_params['ki'] -= learning_rate * grads[1]
    pid_params['kd'] -= learning_rate * grads[2]

    # Store parameter values
    kp_values.append(pid_params['kp'])
    ki_values.append(pid_params['ki'])
    kd_values.append(pid_params['kd'])

    # Calculate and store the MSE for this epoch
    mse = loss_fn(pid_params['kp'], pid_params['ki'], pid_params['kd'])
    mse_values.append(mse)
    
    # Optionally print or log the new parameters and/or loss
    print(f"Epoch {epoch}: KP={pid_params['kp']}, KI={pid_params['ki']}, KD={pid_params['kd']}, MSE={mse}")


# Convert results to NumPy for plotting
heights_np = jnp.array(heights)
errors_np = jnp.array(errors)


# Visualization of PID parameter updates over epochs
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(kp_values, label='Kp', color='r')
plt.plot(ki_values, label='Ki', color='g')
plt.plot(kd_values, label='Kd', color='b')
plt.xlabel('Epoch')
plt.ylabel('Parameter Value')
plt.title('PID Parameters Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mse_values, label='MSE', color='blue')
plt.xlabel('Epoch')
plt.ylabel('MSE Value')
plt.title('MSE Over Epochs')
plt.legend()

plt.grid(True)
plt.show()


