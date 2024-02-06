import numpy as np
import matplotlib.pyplot as plt
from bathtub import bathtub
import jax.numpy as jnp
import jax
from jax import grad


class PID:

    def __init__(self, Kp, Ki, Kd, setpoint, dt):
        """
        :param Kp: Proportional gain
        :param Ki: Integral gain
        :param Kd: Derivative gain
        :param setpoint: The desired value
        :param dt: Time step
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt

        self.I = 0  # Integral
        self.last_error = 0

        self.kp_history = []
        self.ki_history = []
        self.kd_history = []

    def update(self, error):
        P = self.Kp * error  # Proportional
        self.I += error * self.dt
        D = (error - self.last_error) / self.dt  # Derivative
        self.last_error = error
        return P + (self.Ki * self.I) + (self.Kd * D)

    def cost_function(self, error):
        return jnp.mean(error ** 2)

    def reset(self):
        self.I = 0
        self.last_error = 0

    # def update_k_values(self, epochs, learning_rate):
    #     differential_loss_fn = partial_derivative(setpoint, num_time_steps, disturbance_generator)
    #     gradients_loss = jax.grad(differential_loss_fn, argnums=[0, 1, 2])
    #     gradients_jit = jax.jit(gradients_loss)
    #     mse_values = []
    #
    #     for epoch in range(epochs):
    #         gradients = gradients_jit(self.Kp, self.Ki, self.Kd)
    #         self.Kp -= learning_rate * gradients[0]
    #         self.Ki -= learning_rate * gradients[1]
    #         self.Kd -= learning_rate * gradients[2]
    #
    #         self.kp_history.append(self.Kp)
    #         self.ki_history.append(self.Ki)
    #         self.kd_history.append(self.Kd)
    #
    #         mse = differential_loss_fn(self.Kp, self.Ki, self.Kd)
    #         mse_values.append(mse)
    #
    #         if epoch % 10 == 0:
    #             print(f"Epoch: {epoch}, Loss: {mse}")
    #
    #     return self.Kp, self.Ki, self.Kd
    #



# TODO: Mortaza: skal endres!
def make_loss_function(setpoint, time_sample):
    def calcuate_mse(kp, ki, kd):
        return calculate_mse(kp, ki, kd, setpoint, time_sample)
    return calcuate_mse

def calculate_mse(kp, ki, kd, setpoint, time_sample):
    # TODO: fikse dt på en bedre måte:
    controller = PID(kp, ki, kd, setpoint, 1)

    # Reset plant
    plant = bathtub(A=10, C=0.1, H_0=setpoint)

    acc_error = 0.0

    for step in range(time_sample):
        noise = disturbance_generator(x1=-0.01, x2=0.01)  # Noise
        error = plant.get_error()
        controller_output = controller.update(error)
        plant.update(U=controller_output, D=noise)
        acc_error += error ** 2

    mse = acc_error / time_sample
    return mse


def disturbance_generator(x1=-0.01, x2=0.01):
    return np.random.uniform(x1, x2)


def partial_derivative(setpoint, num_steps, disturbance_generator):
    def loss(kp, ki, kd):
        return calculate_mse(setpoint, num_steps)

    return loss


def run_program():
    # Instantiate parameters
    num_epochs = 100
    num_time_steps = 100
    learning_rate = 0.01
    time_sample = 1
    pid_params = {"kp": 1, "ki": 2, "kd": 1.3}

    # bathtub
    setpoint = 100  # aka. Initial_height

    # TODO: skal endres
    # Prepare the loss function for gradients
    loss_fn = calculate_mse(setpoint, time_sample)

    # Compute gradients of the loss function w.r.t. PID parameters
    grad_loss_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    grad_jit = jax.jit(grad_loss_fn)

    mse_history = []
    kp_values = []
    ki_values = []
    kd_values = []

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
        mse_history.append(mse)

        # Optionally print or log the new parameters and/or loss
        print(f"Epoch {epoch}: KP={pid_params['kp']}, KI={pid_params['ki']}, KD={pid_params['kd']}, MSE={mse}")


    # TODO: convert to jax.np array before plotting


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
    plt.plot(mse_history, label='MSE', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Value')
    plt.title('MSE Over Epochs')
    plt.legend()

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
   run_program()
