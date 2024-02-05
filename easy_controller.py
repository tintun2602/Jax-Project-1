import numpy as np
import matplotlib.pyplot as plt
from bathtub import bathtub
import jax.numpy as jnp
import jax
from jax import grad 


class PID:

    ## TODO: Remove the setpoint from the PID class and pass only error as an argument to the update method!
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
        
        self.I = 0 # Integral
        self.last_error = 0

        self.kp_history = []
        self.ki_history = []
        self.kd_history = []

    def update(self, error):
        P = self.Kp * error # Proportional
        self.I += error * self.dt
        D = (error - self.last_error) / self.dt # Derivative
        self.last_error = error
        return P + (self.Ki * self.I) + (self.Kd * D)

    def cost_function(self, error):
        return jnp.mean(error ** 2)
    
    def reset(self):
        self.I = 0
        self.last_error = 0

    def update_k_values(self, epochs, learning_rate):   
        differential_loss_fn = partial_derivative(setpoint, num_time_steps, disturbance_generator)
        gradients_loss = jax.grad(differential_loss_fn, argnums=[0, 1, 2])
        gradients_jit = jax.jit(gradients_loss)
        mse_values = []
        
        for epoch in range(epochs):
            gradients = gradients_jit(self.Kp, self.Ki, self.Kd)
            self.Kp -= learning_rate * gradients[0]
            self.Ki -= learning_rate * gradients[1]
            self.Kd -= learning_rate * gradients[2]

            self.kp_history.append(self.Kp)
            self.ki_history.append(self.Ki)
            self.kd_history.append(self.Kd)

            mse = differential_loss_fn(self.Kp, self.Ki, self.Kd)
            mse_values.append(mse)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {mse}")

        return self.Kp, self.Ki, self.Kd

def calculate_mse(self, setpoint, num_steps):
    controller = PID(Kp=self.Kp, Ki=self.Ki, Kd=self.Kd, setpoint=setpoint, dt=self.dt)
    plant = bathtub(A=10, C=0.1, H_0=setpoint)
    acc_error = 0.0

    for step in range(num_steps):
        noise = disturbance_generator(x1=-0.01, x2=0.01) # Noise
        error = plant.get_error()
        controller_output = controller.update(error)
        plant.update(U=controller_output, D=noise)
        acc_error += error ** 2 
        
    mse = acc_error / num_steps
    return mse

def disturbance_generator(x1=-0.01, x2=0.01):
    return np.random.uniform(x1, x2)


def partial_derivative(setpoint, num_steps, disturbance_generator):
    def loss(kp, ki, kd):
        return calculate_mse(setpoint, num_steps)
    return loss

if __name__ == "__main__":
    num_epochs = 100  
    num_time_steps = 100  
    learning_rate = 0.01
    time_sample = 1

    setpoint = 100 # aka. intitial_height
    pid_params = {"Kp" : 1, "Ki" : 2, "Kd" : 1.3}


    # initial PID parameter values
    pid = PID(Kp=pid_params["Kp"], Ki=pid_params["Ki"], Kd=pid_params["Kd"], setpoint=setpoint, dt=time_sample)
    # optimized PID parameter values
    pid.update_k_values(epochs=num_epochs, learning_rate=learning_rate)

    water_height_history = []
    controller_output_history = []

    # Visualization of PID parameter updates over epochs
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(pid.kp_history, label='Kp', color='r')
    plt.plot(pid.ki_history, label='Ki', color='g')
    plt.plot(pid.kd_history, label='Kd', color='b')
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