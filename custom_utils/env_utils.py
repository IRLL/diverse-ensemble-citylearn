from typing import Any, List, Mapping, Union
from pathlib import Path
import numpy as np
import pandas as pd

from citylearn.citylearn import CityLearnEnv
from citylearn.cost_function import CostFunction


class CityLearnEnvCustomReward(CityLearnEnv):
    """ Differs from CityLearnEnv only in the reward function"""

    def __init__(self, schema: Union[str, Path, Mapping[str, Any]], **kwargs):
        self.total_net_load_history = []
        self.total_net_load_history_no_storage = []
        self.prices = None
        self.prices_noop = None
        self.emissions = None
        self.emissions_noop = None
        super().__init__(schema, **kwargs)

    def reset(self):
        self.total_net_load_history = []
        self.total_net_load_history_no_storage = []
        # cumulative costs:
        self.prices = [0.0 for _ in self.buildings]
        self.prices_noop = [0.0 for _ in self.buildings]
        self.emissions = [0.0 for _ in self.buildings]
        self.emissions_noop = [0.0 for _ in self.buildings]
        return super().reset()

    @property
    def observations(self) -> List[List[float]]:
        obss = super().observations
        net_loads = [ob[23] for ob in obss]
        net_loads_no_storage = [ob[20]-ob[21] for ob in obss]
        self.total_net_load_history.append(sum(net_loads))
        self.total_net_load_history_no_storage.append(sum(net_loads_no_storage))
        return obss

    def get_reward(self) -> List[float]:
        """Calculate agent(s) reward(s) using :attr:`reward_function`.

        Returns
        -------
        reward: List[float]
            Reward for current observations. If `central_agent` is True, `reward` is a list of length = 1 else, `reward` has same length as `buildings`.
        """
        assert self.central_agent is not True
        self.reward_function.carbon_emission = [b.net_electricity_consumption_emission[self.time_step] for b in
                                                self.buildings]
        self.reward_function.electricity_price = [b.net_electricity_consumption_price[self.time_step] for b in
                                                  self.buildings]

        net_demands_no_storage = []
        for b in self.buildings:
            net_demands_no_storage.append(b.net_electricity_consumption[self.time_step] -
                                          b.electrical_storage_electricity_consumption[self.time_step])

        self.reward_function.electricity_price_without_storage = [
            b.pricing.electricity_pricing[self.time_step] * net_dem
            for b, net_dem in zip(self.buildings, net_demands_no_storage)]
        self.reward_function.carbon_emission_without_storage = [
            max(0.0, b.carbon_intensity.carbon_intensity[self.time_step] * net_dem)
            for b, net_dem in zip(self.buildings, net_demands_no_storage)]

        if len(self.total_net_load_history) > 1:
            ramping = np.abs(self.total_net_load_history[-1] - self.total_net_load_history[-2])
            ramping_without_storage = np.abs(
                self.total_net_load_history_no_storage[-1] - self.total_net_load_history_no_storage[-2])
            self.reward_function.ramping_delta = ramping_without_storage - ramping
        else:
            self.reward_function.ramping_delta = 0.0

        netload_window = self.total_net_load_history[-730:]
        netload_no_storage_window = self.total_net_load_history_no_storage[-730:]
        load_factor = 1 - np.mean(netload_window) / np.max(netload_window)
        load_factor_no_storage = 1 - np.mean(netload_no_storage_window) / np.max(netload_no_storage_window)
        self.reward_function.load_factor_delta = load_factor_no_storage - load_factor

        reward = self.reward_function.calculate()
        return reward

    def update_variables(self):
        self.net_electricity_consumption.append(sum([b.net_electricity_consumption[self.time_step] for b in self.buildings]))
        for b_i, b in enumerate(self.buildings):
            assert self.time_step == b.time_step, f"self.time_step={self.time_step} but b.time_step={b.time_step}"
            self.prices[b_i] += max(0.0, b.net_electricity_consumption_price[self.time_step])
            self.prices_noop[b_i] += max(0.0, b.net_electricity_consumption_without_storage_price[self.time_step])
            self.emissions[b_i] += b.net_electricity_consumption_emission[self.time_step]
            self.emissions_noop[b_i] += b.net_electricity_consumption_without_storage_emission[self.time_step]


    def evaluate(self):
        """Only applies to the CityLearn Challenge 2022 setup."""
        # normalized costs
        price_costs = [price / price_noop for price, price_noop in zip(self.prices, self.prices_noop)]
        emission_costs = [emission / emission_noop for emission, emission_noop in zip(self.emissions, self.emissions_noop)]
        total_price_cost = sum(self.prices)/sum(self.prices_noop)
        total_emission_cost = sum(self.emissions)/sum(self.emissions_noop)

        ramping_cost = CostFunction.ramping(self.net_electricity_consumption)[-1] / \
            CostFunction.ramping(self.net_electricity_consumption_without_storage)[-1]
        load_factor_cost = CostFunction.load_factor(self.net_electricity_consumption)[-1] / \
            CostFunction.load_factor(self.net_electricity_consumption_without_storage)[-1]
        grid_cost = np.mean([ramping_cost, load_factor_cost])

        return price_costs, emission_costs, total_price_cost, total_emission_cost, grid_cost
