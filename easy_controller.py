import numpy as np
import matplotlib.pyplot as plt
from bathtub import bathtub
from jax import jit, grad
import jax.numpy as jnp


class PID:

    # inti metode
    def __init__(self, Kp, Ki, Kd, setpoint, dt):
      

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.I = 0

        self.setpoint = setpoint
        self.dt = dt

        self.last_error = 0

    def update(self, current_value):
        error = self.setpoint - current_value
        P = self.Kp * error
        self.I += error * self.dt
        D = (error - self.last_error) / self.dt
        self.last_error = error
        return P + (self.Ki * self.I) + (self.Kd * D)


    def cost_function(self, setpoint, actual):
        error = setpoint - actual
        return jnp.mean(error ** 2)
    
    def update_k_values(self, setpoint, epochs, learning_rate):
        loss_grad = grad(self.cost_function)
        for epoch in range(epochs):
            bathtub_plant = bathtub(A=10, C=0.1, H_0=setpoint)
            for step in range(100):
                current_height = bathtub_plant.state
                controller_output = self.update(current_value=current_height)
                bathtub_plant.update(U=controller_output, D=disturbance_generator())

            actual_height = jnp.array(bathtub_plant.state, dtype=jnp.float32)
            setpoint_jax = jnp.array(setpoint, dtype=jnp.float32)

            mse = self.cost_function(setpoint_jax, actual_height)
            grad_value = loss_grad(setpoint_jax, actual_height)

            # Update Kp, Ki, and Kd parameters using gradient descent
            self.Kp -= learning_rate * grad_value[0]  # Update Kp
            self.Ki -= learning_rate * grad_value[1]  # Update Ki
            self.Kd -= learning_rate * grad_value[2]  # Update Kd

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {mse}")

        return self.Kp, self.Ki, self.Kd

    def visualizePlot(self, water_height_history, output, controller_parameters, errors_np, heights_np, kp_values, ki_values, kd_values, num_epochs):

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        axs[0].plot(errors_np)
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Mean Squared Error')
        axs[0].set_title('Mean Squared Error Over Time')
        axs[0].set_xlim(0, num_epochs)

        axs[1].plot(heights_np)
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Water Height')
        axs[1].set_title('Water Height Over Time')
        axs[1].set_xlim(0, num_epochs)

        axs[2].plot(kp_values, label='Kp')
        axs[2].plot(ki_values, label='Ki')
        axs[2].plot(kd_values, label='Kd')
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('PID Parameters')
        axs[2].set_title('PID Parameters Over Time')
        axs[2].set_xlim(0, num_epochs)
        axs[2].legend()

        plt.tight_layout()
        plt.show()


def disturbance_generator(x1=-0.01, x2=0.01):
    return np.random.uniform(x1, x2)


if __name__ == "__main__":
    goal_height = 100
    epochs = 100
    time_sample = 1

    Kp = 1  # Proposjonal konstanten
    Ki = 2  # Fjerner feil som er konstant
    Kd = 1.3  # Demper ossilasjon (svigninger fra v til h)

    pid = PID(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=goal_height, dt=time_sample)

    # Optimize PID gains
    optimized_Kp, optimized_Ki, optimized_Kd = pid.update_k_values(setpoint=np.array([goal_height]), epochs=100, learning_rate=0.01)

    print(f"Optimized Kp: {optimized_Kp}, Optimized Ki: {optimized_Ki}, Optimized Kd: {optimized_Kd}")

    bathtub_plant = bathtub(A=10, C=0.1, H_0=goal_height)

    water_height_history = []
    controller_output_history = []

    pid = PID(Kp=optimized_Kp, Ki=optimized_Ki, Kd=optimized_Kd, setpoint=goal_height, dt=time_sample)

    for dt in range(100):
        water_height_history.append(bathtub_plant.state)

        current_height = bathtub_plant.state
        controller_output = pid.update(current_value=current_height)
        bathtub_plant.update(U=controller_output, D=disturbance_generator())

        controller_output_history.append(controller_output)


    # Calculate errors, heights, and PID parameter values over epochs
    errors_np = np.array([goal_height - height for height in water_height_history])
    heights_np = np.array(water_height_history)
    kp_values = np.array([optimized_Kp] * len(water_height_history))
    ki_values = np.array([optimized_Ki] * len(water_height_history))
    kd_values = np.array([optimized_Kd] * len(water_height_history))
    num_epochs = len(water_height_history)

    pid.visualizePlot(
        water_height_history=water_height_history,
        output=controller_output_history,
        controller_parameters=(optimized_Kp, optimized_Ki, optimized_Kd, time_sample),
        errors_np=errors_np,
        heights_np=heights_np,
        kp_values=kp_values,
        ki_values=ki_values,
        kd_values=kd_values,
        num_epochs=num_epochs
    )