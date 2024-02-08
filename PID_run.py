import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from plants import BathtubPlant
from utils import disturbance_generator


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
def make_loss_function(time_sample, plant):
    def loss_fn(kp, ki, kd):
        return calculate_mse(kp, ki, kd, time_sample, plant)
    return loss_fn

def calculate_mse(kp, ki, kd, time_sample, plant):
    # TODO: fikse dt på en bedre måte
    controller = PID(kp, ki, kd, 1)
    error_history_over_one_epoch = []
    print(plant.get_error())

    for step in range(time_sample):

        noise = disturbance_generator(x1=-0.01, x2=0.01)  # Noise
        error = plant.get_error()
        controller_output = controller.update(error)
        plant.update_state(controller_output, noise)
        error_history_over_one_epoch.append(error)

    print(error_history_over_one_epoch)
    squared_errors = jnp.square(jnp.array(error_history_over_one_epoch))
    mse = jnp.mean(squared_errors)
    print("mean sqaure error: ", mse)
    return mse


def run_program():
    # Instantiate parameters
    num_epochs = 1000
    num_time_steps = 100
    learning_rate = 0.01
    pid_params = {"kp": 1.5, "ki": 1.8, "kd": 2.5}

    # bathtub
    setpoint = 100  # aka. Initial_height

    plant = BathtubPlant(setpoint, 10, 0.01)

    # Prepare the loss function for gradients
    loss_fn = make_loss_function(num_time_steps, plant)

    # Compute gradients of the loss function w.r.t. PID parameters
    grad_loss_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_jit = jax.jit(grad_loss_fn)

    mse_history = []
    kp_history = []
    ki_history = []
    kd_history = []

    for epoch in range(num_epochs):
        # Compute gradients

        print(f"pid_params ${pid_params}")
        grads = grad_jit(pid_params['kp'], pid_params['ki'], pid_params['kd'])


        # Update PID parameters
        pid_params['kp'] -= learning_rate * grads[0]
        pid_params['ki'] -= learning_rate * grads[1]
        pid_params['kd'] -= learning_rate * grads[2]

        print(f"pid_params 2 ${pid_params['kp']} and ${pid_params['ki']} and ${pid_params['kd']}")

        # Store parameter values
        kp_history.append(pid_params['kp'])
        ki_history.append(pid_params['ki'])
        kd_history.append(pid_params['kd'])

        # Calculate and store the MSE for this epoch
        mse = loss_fn(pid_params['kp'], pid_params['ki'], pid_params['kd'])
        print(f"mse: {mse}")
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
