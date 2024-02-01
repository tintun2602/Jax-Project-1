import numpy as np
import matplotlib.pyplot as plt
from bathtub import bathtub



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

        self.P = 0
        self.I = 0
        self.D = 0

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
        self.P = self.Kp * error
        self.I += error * self.dt
        self.D = (error - self.last_error) / self.dt
        self.last_error = error
        return self.P + (self.Ki * self.I) + (self.Kd * self.D)      

    def update2(self, current_value):
        """
        Denne metoden oppdaterer ikke PID variablene
        Calculates and returns the PID output based on the current value.
        """

        # Calculate the error
        error = self.setpoint - current_value

        # Calculate the proportional term
        proportional = self.Kp * error

        # Calculate the integral term
        self.integral += error * self.sample_time
        integral = self.Ki * self.integral

        # Calculate the derivative term (using a simple approximation)
        derivative = self.Kd * (error - self.last_error) / self.sample_time

        # Update the last error for the next iteration
        self.last_error = error

        # Calculate the total PID output
        output = proportional + integral + derivative

        return output


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
    #  Variabler
    goal_height = 100
    
    Kp = 10   # 
    Ki = 2    # Fjerner feil som er konstant
    Kd = 1.3    # demper ossilasjon (svingnigner fra v til h)
    time_sample = 1

    bathtub_plant = bathtub(A = 10, C = 0.1, H_0 = goal_height)
    pid = PID(Kp = Kp, Ki = Ki, Kd = Kd, setpoint = goal_height, dt = time_sample)

    water_height_history = []
    controller_output_history = []

    for dt in range(20):
        water_height_history.append(bathtub_plant.state)

        current_height = bathtub_plant.state
        controller_output = pid.update(current_value=current_height)
        bathtub_plant.update(U = controller_output, D = disturbance_generator()) 

        controller_output_history.append(controller_output)

    pid.update(0.5)
    pid.visualizePlot(
        output = controller_output_history,
        water_height_history = water_height_history,
        controller_parameters=(Kp, Ki, Kd, time_sample))

