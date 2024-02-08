
import numpy as np

import matplotlib.pyplot as plt

from neural_network import neural_network
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






if __name__ == "__main__":

    # Step 1: Initialize the BathtubPlant
    initial_level = 1.0  # Initial water level in meters
    area = 10.0  # Cross-sectional area of the bathtub in square meters
    drain_area = 0.1  # Area of the drain in square meters


    """
    For Bathtub Plant include Dt
    
    For CournotCompetition exclude Dt
    For CournotCompetition exclude Dt
    
    """
    
    #plant = BathtubPlant(initial_level, area, drain_area, ) # Includes Dt
    plant = CournotCompetition(1.0, 5.0, 1.0) # Exclude Dt
    features, targets = generate_pid_training_data(1.0, n_samples=1000)  # Example setpoint and samples
    

    # Step 4: Run the Control Loop
    setpoint = 1.0  # Desired water level in meters
    dt = 1.0  # Time step in seconds
    time_steps = 20

    # create nn controller
    nn = neural_network()


    
    trained_params, mse_history = nn.train_neural_network(features, targets, epochs=100, lrate=0.1)

    nn.control_loop(plant, trained_params, setpoint, time_steps, dt)
    
    # Plot the MSE history after training
    plt.figure(figsize=(10, 6))
    plt.plot(mse_history, label='Training MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Mean Squared Error Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

