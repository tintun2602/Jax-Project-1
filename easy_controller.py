import numpy as np
import matplotlib.pyplot as plt
from bathtub import bathtub
from jax import jit, grad
import jax.numpy as jnp


class PID:

    # inti metode
    def __init__(self, Kp, Ki, Kd, setpoint, dt):
        """
        
        PID parameters: Kp, Ki, Kd
        Setpoint: desired value
        dt: delta time

        Proportioal (P): 
        The proportional response can be adjusted by multiplying the error by a constant known as the Proportional Gain, Kp. 
        A high Kp will increase the speed of the system response but can lead to an unstable system with overshoot.

        Integral (I):
        This component is concerned with the accumulation of past errors. 
        If the error has been present for a prolonged period, the integral term increases, thereby increasing the controller output and driving the error towards zero.
        : (1 / 100 of P)

        Derivative (D):
        This component is a prediction of future error, based on its rate of change. 
        It provides a damping force that reduces overshoot and improves system stability.
        : (1 x 10 of P)

        """

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.I = 0

        self.setpoint = setpoint
        self.dt = dt

        # Når pid controllen setters på antar vi at det ikke er noen feil
        self.last_error = 0

    def update(self, current_value):
        """
        NOTE: denne metoden oppdaterer PID variabelene
        error: error value
        return: PID output (controller signal)
        """

        error = self.setpoint - current_value
        P = self.Kp * error
        self.I += error * self.dt
        D = (error - self.last_error) / self.dt
        self.last_error = error
        return P + (self.Ki * self.I) + (self.Kd * D)

    # def objective(self, setpoint, actual):
    #     """
    #     Calculates the mean sqared error
    #
    #     :param:
    #     """
    #     error = setpoint - actual
    #     return np.mean(error ** 2)

    # def update_k_values(self, loss_grad, learning_rate):
    #     """Updates PID gains using gradient descent."""
    #
    #     # Unpack gradients
    #     Kp_grad, Ki_grad, Kd_grad = loss_grad
    #
    #     # Update gains with gradient descent
    #     self.Kp -= learning_rate * Kp_grad
    #     self.Ki -= learning_rate * Ki_grad
    #     self.Kd -= learning_rate * Kd_grad
    #
    #     return Kp, Ki, Kd

    def cost_function(self, setpoint, actual):
        """Cost function to minimize."""
        error = setpoint - actual
        return jnp.mean(error ** 2)

    def update_k_values(self, setpoint, epochs, learning_rate):
        """Optimize PID gains using gradient descent."""
        # print(self.cost_function)
        loss_grad = grad(self.cost_function)

        for epoch in range(epochs):
            bathtub_plant = bathtub(A=10, C=0.1, H_0=setpoint)

            for _ in range(100):
                current_height = bathtub_plant.state
                controller_output = self.update(current_value=current_height)
                bathtub_plant.update(U=controller_output, D=disturbance_generator())

            # actual_height = bathtub_plant.state.astype(jnp.float32)  # Convert to float32
            # setpoint_jax = setpoint.astype(jnp.float32)  # Convert to float32

            actual_height = jnp.array(bathtub_plant.state, dtype=jnp.float32)
            setpoint_jax = jnp.array(setpoint, dtype=jnp.float32)

            loss = self.cost_function(setpoint_jax, actual_height)
            loss_grad_values = loss_grad(setpoint_jax, actual_height)

            # Unpack gradients
            # Kp_grad, Ki_grad, Kd_grad = jnp.ravel(loss_grad_values)
            Kp_grad, Ki_grad, Kd_grad = jnp.ravel(jnp.reshape(loss_grad_values, (1, -1)))

            # Update gains with gradient descent
            self.Kp -= learning_rate * Kp_grad
            self.Ki -= learning_rate * Ki_grad
            self.Kd -= learning_rate * Kd_grad

            # Visualize progress every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss}")

        return self.Kp, self.Ki, self.Kd

    def visualizePlot(self, output, water_height_history, controller_parameters):
        Kp, Ki, Kd, dt = controller_parameters

        mse_np = np.array(output)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(mse_np)
        axs[0].set_xlabel('dt')
        axs[0].set_ylabel('Controller Output')
        axs[0].set_title('Controller Output')
        axs[0].grid(True)

        # Please add these notes copilot
        axs[0].text(0.5, 0.9, f'Kp={Kp}, Ki={Ki}, Kd={Kd}, dt={dt}',
                    fontsize=10, color='red', transform=axs[0].transAxes, bbox=dict(facecolor='white', alpha=0.8))

        axs[1].plot(water_height_history)
        axs[1].set_xlabel('dt')
        axs[1].set_ylabel('Water Height')
        axs[1].set_title('Water Height')
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()


def disturbance_generator(x1=-0.01, x2=0.01):
    return np.random.uniform(x1, x2)


if __name__ == "__main__":
    goal_height = 100
    epochs = 100
    time_sample = 1

    Kp = 10  # Proposjonal konstanten
    Ki = 2  # Fjerner feil som er konstant
    Kd = 1.3  # Demper ossilasjon (svingnigner fra v til h)

    pid = PID(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=goal_height, dt=time_sample)

    # Optimize PID gains
    optimized_Kp, optimized_Ki, optimized_Kd = pid.update_k_values(setpoint=np.array([goal_height]), epochs=100, learning_rate=0.01)





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







    # water_height_history = []
    # controller_output_history = []
    # # Optimization loop
    # for _ in range(epochs):
    #     bathtub_plant = bathtub(A=10, C=0.1, H_0=goal_height)
    #
    #     for dt in range(100):
    #         water_height_history.append(bathtub_plant.state)
    #
    #         current_height = bathtub_plant.state
    #         controller_output = pid.update(current_value=current_height)
    #         bathtub_plant.update(U=controller_output, D=disturbance_generator())
    #
    #         controller_output_history.append(controller_output)
    #
    # loss = self.objective()

    pid.visualizePlot(
        output=controller_output_history,
        water_height_history=water_height_history,
        controller_parameters=(optimized_Kp, optimized_Ki, optimized_Kd, time_sample))

# TODO: Create optimization loop -  simple implementation is already done
# - > Starting from for _ in range(epochs)


# # Optimization loop
# for _ in range(num_iterations):
#     # Simulate with current gains
#     H_sim = ...  # Initialize H_sim
#     H_prev = H_sim  # Initialize previous state for derivative
#     for t in range(time_horizon):
#         U = controller(H_sim, H_desired, Kp, Ki, Kd)
#         H_sim = step(H_sim, U, D[t])
#         H_prev = H_sim  # Update previous state for derivative

#     # Evaluate objective function
#     loss = objective(H_sim, H_desired)

#     # Calculate gradients with JAX
#     loss_grad = grad(objective)(H_sim, H_desired)

#     # Update gains using optimizer (replace with your chosen algorithm)
#     Kp, Ki, Kd = update_gains(Kp, Ki, Kd, loss_grad, learning_rate)


# # def loss_grad(self, setpoint, actual):
# #         """Calculates the gradient of the loss function with respect to the PID gains."""

# #         # Calculate error
# #         error = setpoint - actual

# #         # Calculate gradients
# #         Kp_grad = np.mean(-2 * error * self.P)
# #         Ki_grad = np.mean(-2 * error * self.I)
# #         Kd_grad = np.mean(-2 * error * self.D)

# #         return Kp_grad, Ki_grad, Kd_grad
