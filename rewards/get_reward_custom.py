# -*- coding: utf-8 -*-
# @Time    : 2022/10/21 10:44 AM
# @Author  : Zhihu Yang
from typing import List
import numpy as np

###########################################################################
#####                Specify your reward function here                #####
###########################################################################

with_grid_costs = False
if with_grid_costs:
    reward_descriptor = "-withGridCosts"
else:
    reward_descriptor = "-onlyIndivCosts"

# total_prices_without_storage = np.array(
#     [2250.369380385991, 1965.6434880131649, 1315.517106598308, 1706.6579301365518, 1540.7116858494592])
# total_emissions_without_storage = np.array(
#     [1117.2326344334315, 1042.943169045011, 707.0223401835485, 967.5120824492801, 791.4019636261056])
#
# total_ramping_without_storage = 14807.7076
# total_load_factor_without_storage = 7619.0070  # 1 - avg/max over a window of past 730 net_electricity_consumption values


def get_reward(emission: List[float], price: List[float], agent_count: int,
               emission_without_storage: List[float], price_without_storage: List[float],
               ramping_delta: float, load_factor_delta: float
               ) -> dict:
    """CityLearn Challenge user reward calculation.

    Parameters
    ----------
    electricity_consumption: List[float]
        List of each building's/total district electricity consumption in [kWh].
    carbon_emission: List[float]
        List of each building's/total district carbon emissions in [kg_co2].
    electricity_price: List[float]
        List of each building's/total district electricity price in [$].

    Returns
    -------
    rewards: dict
    """

    assert emission_without_storage is not None and price_without_storage is not None, "check inputs!"

    emission = np.array(emission).clip(min=0)
    price = np.array(price).clip(min=0)
    emission_without_storage = np.array(emission_without_storage).clip(min=0)
    price_without_storage = np.array(price_without_storage).clip(min=0)

    emission_reward = emission_without_storage - emission
    price_reward = price_without_storage - price

    reward = {"emission": emission_reward, "price": price_reward}

    if with_grid_costs:
        ramping_reward_per_agent = ramping_delta / agent_count
        load_factor_reward_per_agent = load_factor_delta / agent_count
        reward.update({"ramping": ramping_reward_per_agent, "load_factor": load_factor_reward_per_agent})

    return reward
