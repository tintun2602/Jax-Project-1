import numpy as np
import matplotlib.pyplot as plt
from bathtub import bathtub

"""

Control a bathtub system using PID controller with Ziegler-Nichols tuning method

"""

class PID:
    def __init__(self, Kp, Ki, Kd, setpoint, dt):
        # Initialization of PID controller

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.I = 0
        self.setpoint = setpoint
        self.dt = dt
        self.last_error = 0

    def update(self, current_value):
        # Update PID controller

        error = self.setpoint - current_value
        P = self.Kp * error
        self.I += error * self.dt
        D = (error - self.last_error) / self.dt
        self.last_error = error
        return P + (self.Ki * self.I) + (self.Kd * D)

def disturbance_generator(x1=-0.01, x2=0.01):
    # Generate disturbances
    return np.random.uniform(x1, x2)

def calculate_mse(setpoint, plant_output):
    # Calculate Mean Squared Error (MSE)
    errors = np.array(setpoint) - np.array(plant_output)
    mse = np.mean(errors ** 2)
    return mse

def ziegler_nichols_tuning(setpoint, disturbances, dt):
    # Use Ziegler-Nichols method to determine initial PID gains
    Kp = 0.2  # Adjust the value as needed
    Ki = 1.2 / dt
    Kd = 0.075 * dt


    # Simulate the system with the initial gains
    plant_output = []
    pid = PID(Kp, Ki, Kd, setpoint, dt)
    for _ in range(len(disturbances)):
        current_height = plant_output[-1] if plant_output else setpoint
        controller_output = pid.update(current_height)
        plant_output.append(current_height + controller_output + disturbance_generator())

    # Calculate MSE for the initial gains
    mse = calculate_mse(setpoint, plant_output)

    return Kp, Ki, Kd, mse

def main():
    goal_height = 100
    time_sample = 1
    disturbances = [disturbance_generator() for _ in range(100)]

    # Use Ziegler-Nichols to determine initial PID gains
    Kp, Ki, Kd, initial_mse = ziegler_nichols_tuning(goal_height, disturbances, time_sample)

    print(f"Initial PID Gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    print(f"Initial MSE: {initial_mse}")

    bathtub_plant = bathtub(A=10, C=0.1, H_0=goal_height)
    pid = PID(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=goal_height, dt=time_sample)

    water_height_history = []
    controller_output_history = []

    for _ in range(100):
        water_height_history.append(bathtub_plant.state)
        current_height = bathtub_plant.state
        controller_output = pid.update(current_height)
        bathtub_plant.update(U=controller_output, D=disturbance_generator())
        controller_output_history.append(controller_output)

    final_mse = calculate_mse([goal_height] * len(water_height_history), water_height_history)

    print(f"Final MSE with Initial Gains: {final_mse}")

    mse_np = np.array(controller_output_history)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(mse_np)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Controller Output')
    axs[0].set_title('Controller Output')
    axs[0].grid(True)

    axs[1].plot(water_height_history)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Water Height')
    axs[1].set_title('Water Height')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
