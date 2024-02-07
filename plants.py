import jax.numpy as jnp

class CournotCompetition:
    def __init__(self, initial_production, p_max, rival_production):
        self.q1 = max(initial_production, 0)
        self.p_max = p_max
        self.q2 = rival_production

    def get_state(self):
        total_production = self.q1 + self.q2
        price = self.p_max - total_production
        profit = self.q1 * price
        return profit

    def update_state(self, control_signal, dt):
        self.q1 += control_signal * dt


class BathtubPlant:
    def __init__(self, initial_level, area, drain_area, g=9.8):
        self.water_level = initial_level
        self.A = area
        self.C = drain_area
        self.g = g
        self.initial_height = initial_level

    def get_state(self):
        return self.water_level

    def update_state(self, control_signal, D, dt):
        self.V = jnp.sqrt(2 * self.g * self.water_level)
        self.Q = self.V * self.C
        delta_B = (control_signal + D - self.Q) * dt
        delta_H = delta_B / self.A
        self.water_level += delta_H

    def reset(self):
        self.water_level = self.initial_height