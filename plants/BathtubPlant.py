from plants.BasePlant import BasePlant
import jax.numpy as jnp


class BathubPlant(BasePlant):
    def __init__(self, A, C, H_0):
        """
        FOR EVERY TIMESTEP
        :param A: Cross-sectional area of the bathtub from top to bottom
        :param C: Cross-sectional area of th drain
        :param H_O: Initial height of the water in the bathtub

        self.A = Constant cross-sectional area of the bathtub from top to bottom
        self.C = Constant cross-sectional area of th drain
        self.H = Height of water in the bathtub
        self.B = Volume of the bathtub
        self.V = Velocity of water exiting through the drain
        self.Q = Flow rate of exiting water trough the drain, how much water leaves the bathtub
        self.g = Gravitational constant
        """
        self.g = 9.8

        self.A = A  # constant does not change
        self.C = C  # constant does not change
        self.initial_height = H_0
        self.state = H_0  # Initial height of the water in the bathtub
        self.B = A * H_0  # Initial volume of the bathtub
        self.V = jnp.sqrt(2 * self.g * self.state)  # Initial velocity of the bathtub
        self.Q = self.V * self.C  # Initial flow rate of the drain in the bathtub

    def update(self, U, D):
        """
        This function is used to update the bathtub volume and height

        :param U: The output of the controller, amount of water put into the bathtub at each timestep
        :param D: A random noise / disturbance amount of water that varies with each timestep
        :return:
        """
        # Calculating and updating velocity (V) and flow rate of exiting water (Q)
        self.V = jnp.sqrt(2 * self.g * self.state)
        self.Q = self.V * self.C

        # Calculating the change in volume and height of the water in bathtub
        delta_B = U + D - self.Q
        delta_H = delta_B / self.A

        # Updating the volume and the height of the water in bathtub for every timestep (1 sek)
        self.state = self.state + delta_H
        self.B = self.B + delta_B

    def reset(self):
        """
        This function is used to reset the plant to its initial state
        :return: none
        """
        self.state = self.initial_height
