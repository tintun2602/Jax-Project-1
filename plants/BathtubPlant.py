import math
import jax.numpy as jnp


class BathubPlant:
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

        self.A = A # constant does not change
        self.C = C # constant does not change
        self.H = H_0
        self.B = A * H_0    # Initial volume of the bathtub, kanskje dette er unødvendig?, kan bli evt. brukt til å plotte.
        self.V = jnp.sqrt(2 * self.g * self.H)
        self.Q = self.V * self.C  # skal dette også kankje være i timestep funksjonen??

    def timestep(self, U, D):
        """
        This function is used to update the bathtub volume and height

        :param U: The output of the controller, amount of water put into the bathtub at each timestep
        :param D: A random noise / disturbance amount of water that varies with each timestep
        :return:
        """
        # Calculating and updating velocity (V) and flow rate of exiting water (Q)
        self.V = jnp.sqrt(2 * self.g * self.H)
        self.Q = self.V * self.C

        # Calculating the change in volume and height of the water in bathtub
        delta_B = U + D - self.Q
        delta_H = delta_B / self.A

        # Updating the volume and the height of the water in bathtub for every timestep (1 sek)
        self.H = self.H + delta_H
        self.B = self.B + delta_B


if __name__ == '__main__':
    bathtub_plant = BathubPlant(10, 0.01, 200)
    for _ in range(1_00):
        bathtub_plant.timestep(0.01, 0.003)
        print(bathtub_plant.H)

