import random

from ControlSystem import ControlSystem
from controllers.ClassicPIDController import ClassicPIDController
from plants.BathtubPlant import BathubPlant


def main():
    # creating an instance of the bathtub plant
    cross_sectional_area_of_bathtub = 10
    cross_sectional_area_of_bathtub_drain = 0.01
    initial_and_goal_height_of_water_in_bathtub = 100

    bathtub_plant = BathubPlant(
        cross_sectional_area_of_bathtub,
        cross_sectional_area_of_bathtub_drain,
        initial_and_goal_height_of_water_in_bathtub
    )

    # creating an instance of the classical_pid_controller
    classical_pid = ClassicPIDController(kp=1.0, ki=0.1, kd=0.01)

    controll_system = ControlSystem(classical_pid, bathtub_plant)

    epochs = 100
    timesteps = 100

    controll_system.run_control_loop(
        initial_and_goal_height_of_water_in_bathtub,
        epochs,
        timesteps
    )

if __name__ == "__main__":
    main()
