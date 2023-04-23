from typing import List
import numpy as np

###########################################################################
#####                Specify your reward function here                #####
###########################################################################

def get_reward(
        electricity_consumption: List[float],
        carbon_emission: List[float],
        electricity_price: List[float],
        _,
) -> List[float]:
    """CityLearn Challenge user reward calculation.

    Parameters
    ----------
    electricity_consumption: List[float]
        List of each building's/total district electricity consumption in [kWh].
    carbon_emission: List[float], optional
        List of each building's/total district carbon emissions in [kg_co2].
    electricity_price: List[float], optional
        List of each building's/total district electricity price in [$].

    Returns
    -------
    rewards: List[float]
        Agent(s) reward(s) where the length of returned list is either = 1 (central agent controlling all buildings)
        or = number of buildings (independent agent for each building).
    """

    # *********** BEGIN EDIT ***********
    # Replace with custom reward calculation
    building_number = len(electricity_consumption)
    carbon_emission = np.array(carbon_emission).clip(min=0)
    electricity_price = np.array(electricity_price).clip(min=0)
    # electricity_consumption = np.array(electricity_consumption).clip(min=0)
    # sum_electricity_consumption = sum(electricity_consumption)
    # grid_cost = sum_electricity_consumption / 5

    sum_carbon_emission_electricity_price = electricity_price + carbon_emission

    # other_carbon_emission_electricity_price = []
    # for i in range(5):
    #     curr_carbon_emission_electricity_price = copy.deepcopy(sum_carbon_emission_electricity_price).tolist()
    #     del curr_carbon_emission_electricity_price[i]
    #     other_carbon_emission_electricity_price.append(sum(curr_carbon_emission_electricity_price))
    #
    # reward = (sum_carbon_emission_electricity_price * 0.5 + np.array(other_carbon_emission_electricity_price) / 4 * 0.5) * -1 ** 3

    reward = sum_carbon_emission_electricity_price * -1 ** 3
    # ************** END ***************

    return reward
