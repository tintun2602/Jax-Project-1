
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from neural_network import gen_jaxnet_params, jaxnet_train_one_epoch, predict
from pid_controller import generate_pid_training_data
from plants import BathtubPlant, CournotCompetition
from utils import disturbance_generator


def generate_pid_training_data(setpoint, n_samples=1000):
    # Initialize arrays to store PID features and control actions
    features = np.zeros((n_samples, 3))  # P, I, D
    targets = np.zeros((n_samples, 1))   # Control action
    
    # Variables to keep track of integral and previous error for derivative calculation
    integral_error = 0.0
    prev_error = 0.0
    
    for i in range(n_samples):
        # Simulate or calculate the error based on some hypothetical scenarios
        error = np.random.uniform(-1, 1)  # Placeholder for actual error calculation
        integral_error += error
        derivative_error = error - prev_error
        
        # Placeholder for determining the appropriate control action
        # In a real scenario, this would be based on a PID formula or by simulating the plant's response
        control_action = np.random.uniform(-1, 1)  # Placeholder
        
        # Update features and targets
        features[i] = [error, integral_error, derivative_error]
        targets[i] = [control_action]
        
        prev_error = error
    
    return features, targets

def train_neural_network(features, targets, epochs=100, lrate=0.1):
    params = gen_jaxnet_params([3, 10, 1])  # PID features to control action
    mse_history = []  # To store MSE for plotting
    
    for epoch in range(epochs):
        params, mse = jaxnet_train_one_epoch(params, features, targets, lrate)
        mse_history.append(mse)
        print(f'Epoch {epoch}, MSE: {mse}')
    
    return params, mse_history


def control_loop(plant, trained_params, setpoint, time_steps, dt):
    integral_error = 0.0
    prev_error = None  # Will be used to calculate the derivative of error
    error_history = []  # Optional: track error over time for visualization
    control_signal_history = []  # Optional: track control signals over time
    water_level_history = []  # Optional: track water level over time

    for t in range(time_steps):
        current_level = plant.get_state()
        error = setpoint - current_level
        integral_error += error * dt  # Update integral of error
        derivative_error = (error - prev_error) / dt if prev_error is not None else 0.0  # Calculate derivative of error
        prev_error = error  # Update previous error for next iteration
        
        # Use the neural network to predict the control action
        control_action = predict(trained_params, np.array([[error, integral_error, derivative_error]]))
        control_signal = control_action.item()  # Assuming the NN output is a single value
        
        D = disturbance_generator()  # Generate a random disturbance
        # Apply the control signal to the plant
        #plant.update_state(control_signal, D, dt)  # Assuming no external disturbance (D=0)
        plant.update_state(control_signal, D)
        # Optional: Track performance over time
        error_history.append(error)
        control_signal_history.append(control_signal)
        water_level_history.append(current_level)

    print(f'Final water level: {plant.get_state()}')

if __name__ == "__main__":

    # Step 1: Initialize the BathtubPlant
    initial_level = 1.0  # Initial water level in meters
    area = 10.0  # Cross-sectional area of the bathtub in square meters
    drain_area = 0.1  # Area of the drain in square meters
    
    
    # plant = BathtubPlant(initial_level, area, drain_area)
    plant = CournotCompetition(1.0, 5.0, 1.0)
    features, targets = generate_pid_training_data(1.0, n_samples=1000)  # Example setpoint and samples
    
    
    # Step 4: Run the Control Loop
    setpoint = 1.0  # Desired water level in meters
    dt = 1.0  # Time step in seconds
    time_steps = 20
    
    trained_params, mse_history = train_neural_network(features, targets, epochs=100, lrate=0.1)

    control_loop(plant, trained_params, setpoint, time_steps, dt)
    
    # Plot the MSE history after training
    plt.figure(figsize=(10, 6))
    plt.plot(mse_history, label='Training MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Mean Squared Error Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

