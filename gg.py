import numpy as np
import matplotlib.pyplot as plt
from bathtub import bathtub
import jax.numpy as jnp
import jax

class PID:

    def __init__(self, kp, ki, kd, dt):
        """
        :param Kp: Proportional gain
        :param Ki: Integral gain
        :param Kd: Derivative gain
        :param dt: Time step
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.I = 0  # Integral
        self.last_error = 0

        self.kp_history = []
        self.ki_history = []
        self.kd_history = []

    def update(self, error):
        P = self.kp * error  # Proportional
        self.I += error * self.dt
        D = (error - self.last_error) / self.dt  # Derivative
        self.last_error = error
        return P + (self.ki * self.I) + (self.kd * D)



# TODO: Mortaza: skal endres!
def make_loss_function(setpoint, time_sample):
    def loss_fn(kp, ki, kd):
        return calculate_mse(kp, ki, kd, setpoint, time_sample)
    return loss_fn

def calculate_mse(kp, ki, kd, setpoint, time_sample):
    # TODO: fikse dt på en bedre måte
    controller = PID(kp, ki, kd, 1)
    plant = bathtub(A=10, C=0.1, H_0=setpoint)
    error_history_over_one_epoch = []

    for step in range(time_sample):
        noise = disturbance_generator(x1=-0.01, x2=0.01)  # Noise
        error = plant.get_error()
        controller_output = controller.update(error)
        plant.update(U=controller_output, D=noise)
        error_history_over_one_epoch.append(error)

    squared_errors = jnp.square(jnp.array(error_history_over_one_epoch))
    mse = jnp.mean(squared_errors)
    print("mean sqaure error: ", mse)
    return mse

def disturbance_generator(x1=-0.01, x2=0.01):
    return np.random.uniform(x1, x2)


def run_program():
    # Instantiate parameters
    num_epochs = 1000
    num_time_steps = 100
    learning_rate = 0.01
    pid_params = {"kp": 1.5, "ki": 1.8, "kd": 2.5}

    # bathtub
    setpoint = 100  # aka. Initial_height

    # Prepare the loss function for gradients
    loss_fn = make_loss_function(setpoint, num_time_steps)

    # Compute gradients of the loss function w.r.t. PID parameters
    grad_loss_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_jit = jax.jit(grad_loss_fn)

    mse_history = []
    kp_history = []
    ki_history = []
    kd_history = []

    for epoch in range(num_epochs):
        # Compute gradients
        grads = grad_jit(pid_params['kp'], pid_params['ki'], pid_params['kd'])

        # Update PID parameters
        pid_params['kp'] -= learning_rate * grads[0]
        pid_params['ki'] -= learning_rate * grads[1]
        pid_params['kd'] -= learning_rate * grads[2]

        # Store parameter values
        kp_history.append(pid_params['kp'])
        ki_history.append(pid_params['ki'])
        kd_history.append(pid_params['kd'])

        # Calculate and store the MSE for this epoch
        mse = loss_fn(pid_params['kp'], pid_params['ki'], pid_params['kd'])
        mse_history.append(mse)

    # convert to jnp.array
    kp_history = jnp.array(kp_history)
    ki_history = jnp.array(ki_history)
    kd_history = jnp.array(kd_history)
    mse_history = jnp.array(mse_history)

    plot_results(kp_history, ki_history, kd_history, mse_history)

def plot_results(kp_history, ki_history, kd_history, mse_history):
    """
    :param ki_history:
    :param kd_history:
    :param kp_history:
    :param mse_history:
    :return:
    """
    # Visualization of PID parameter updates over epochs
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(kp_history, label='Kp', color='r')
    plt.plot(ki_history, label='Ki', color='g')
    plt.plot(kd_history, label='Kd', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Parameter Value')
    plt.title('PID Parameters Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mse_history, label='MSE', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Value')
    plt.title('MSE Over Epochs')
    plt.legend()

    plt.grid(True)
    plt.show()

if __name__ == "__main__":
   run_program()
