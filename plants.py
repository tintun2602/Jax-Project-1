import jax.numpy as jnp

class CournotCompetition:
    def __init__(self, initial_production, p_max, rival_production, cost_production):
        self.q1 = max(initial_production, 0)
        self.p_max = p_max
        self.q2 = rival_production
        self.cost_production = cost_production
        self.initial_production = initial_production

    def get_state(self):
        total_production = self.q1 + self.q2
        price = self.p_max - total_production
        profit = self.q1 * price - (self.q1 * self.cost_production)
        return profit

    def update_state(self, control_signal, noise):
        self.q2 = self.q2 + noise
        self.q1 += control_signal

    def get_error(self):
        return self.initial_production - self.q1


class BathtubPlant:
    def __init__(self, initial_level, area, drain_area, g=9.8):
        self.water_level = initial_level
        self.A = area
        self.C = drain_area
        self.g = g
        self.initial_height = initial_level

    def get_state(self):
        return self.water_level

    def update_state(self, control_signal, D, dt=1):
        """
        :param dt: dt is 1 sec
        """
        self.V = jnp.sqrt(2 * self.g * self.water_level)
        self.Q = self.V * self.C
        delta_B = (control_signal + D - self.Q) * dt
        delta_H = delta_B / self.A
        self.water_level += delta_H

    def reset_state(self):
        self.water_level = self.initial_height

    def get_error(self):
        return self.initial_height - self.water_level



class TemperatureControlPlant:
    def __init__(self, initial_temp, external_temp, insulation_quality, heating_efficiency):
        """
        Initializes the temperature control model.

        :param initial_temp: Initial temperature of the room (degrees Celsius). This is also the setpoint / goal state

        :param external_temp: External temperature outside the room (degrees Celsius).
        :param insulation_quality: A factor representing the insulation quality of the room,
                                   higher means better insulation (unitless).
        :param heating_efficiency: Efficiency of the heating/cooling system (degrees Celsius
                                   change per unit of control signal).
        """
        self.room_temp = initial_temp
        self.external_temp = external_temp
        self.insulation_quality = insulation_quality
        self.heating_efficiency = heating_efficiency
        self.initial_temp = initial_temp

    def get_state(self):
        """
        Returns the current temperature of the room.
        """
        return self.room_temp

    # TODO: har funk integral??
    def update_state(self, control_signal, dt):
        """
        Updates the room's temperature based on the heating/cooling control signal and the
        external environment over a time step.

        :param control_signal: Control signal for the heating/cooling system. Positive for heating,
                               negative for cooling.
        :param dt: Time step over which to apply the control signal (in seconds).
        """
        # Compute the effect of heating/cooling
        temp_change_due_to_hvac = control_signal * self.heating_efficiency * dt

        # Compute the effect of external temperature
        temp_change_due_to_external = (self.external_temp - self.room_temp) / self.insulation_quality * dt

        # Update the room temperature
        self.room_temp += temp_change_due_to_hvac + temp_change_due_to_external

    def get_error(self):
        return self.initial_temp - self.room_temp