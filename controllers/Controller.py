from abc import ABC, abstractmethod


"""

This is an abstract controller that provides a blueprint for future controller types. 
Which in turn can promote code reusability, extensibility and enforcement of consistency

"""

class Controller(ABC):
    @abstractmethod
    def compute_control_signal(self, error, dt):
        """
        Compute the control signal based on the given error and time interval (dt). 

        :param error: The current error of the system 
        :param dt: The time interval since the last update
        :return: The control signal to be applied
        """
        pass

    @abstractmethod
    def update(self, setpoint, current_value, dt):
        """
        Update the controller's state based on the setpoint, current value, and time interval (dt). 

        :param setpoint: The desired target value
        :param current_value: The current state of the system
        :param dt: The time interval since the last update
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the controller's internal state variables to their initial values.
        """
        pass
