import numpy as np


class TOUPeakReductionAgent:
    """ The Time-of-Use Peak Reduction strategy"""
    def reset(self):
        pass

    def compute_action(self, observation, **kwargs):
        """Get observation return action"""
        soc = observation[22]
        hour = observation[2]
        capacity = 6.4  # all houses have batteries with the same capacities
        if 9 <= hour <= 12:
            action = 2.0 / capacity
        elif (hour >= 18 or hour < 9) and soc > 0.25:
            action = -2.0/capacity
        else:
            action = 0.0
        action = np.array([action])
        return action  # e.g.: array([0.091], dtype=float32)
