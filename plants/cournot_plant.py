import jax.numpy as jnp
from plants.BasePlant import BasePlant

class CournotPlant(BasePlant):
    def __init__(self, cm, pmax, T, q1_0=0.5, q2_0=0.5):
        """
        FOR EVERY TIMESTEP
        :param cm: Marginal cost, the cost to produce each item
        :param pmax: Highest possible price
        :param T: Target profit for each timestep
        :param q1_0: Initial quantity produced by producer 1
        :param q2_0: Initial quantity produced by producer 2

        self.q1: Quantity produced by producer 1
        self.q2: Quantity produced by producer 2
        self.q: Total quantity produced (q1 + q2)
        self.p: Price as a function of total quantity produced (pmax - q)
        self.P1: Profit of producer 1
        self.T: Target profit for each timestep
        self.cm: Marginal cost
        """
        self.q1 = q1_0
        self.q2 = q2_0
        self.q = q1_0 + q2_0
        self.p = pmax - self.q
        self.P1 = q1_0 * (self.p - cm)
        self.T = T
        self.cm = cm

    def update(self, E):
        """
        This function is used to update the Cournot competition model

        :param E: Error, the difference between the target profit and producer 1's profit
        :return:
        """
        # Update quantities produced by producers
        self.q1 += E  # Controller output U is equivalent to ∂q1/∂t
        self.q2 += jnp.random.uniform(0, 0.1)  # Noise/Disturbance D for producer 2

        # Enforce constraints 0 ≤ q1 ≤ 1 and 0 ≤ q2 ≤ 1
        self.q1 = jnp.clip(self.q1, 0, 1)
        self.q2 = jnp.clip(self.q2, 0, 1)

        # Update total quantity produced and price
        self.q = self.q1 + self.q2
        self.p = pmax - self.q

        # Calculate profit for producer 1
        self.P1 = self.q1 * (self.p - self.cm)

    def reset(self):
        """
        This function is used to reset the plant to its initial state
        :return: none
        """
        self.q1 = 0.5
        self.q2 = 0.5
        self.q = 1.0
        self.p = pmax - self.q
        self.P1 = self.q1 * (self.p - self.cm)
