# -*- coding: utf-8 -*-
# @Time    : 2022/10/21 10:42 AM
# @Author  : Zhihu Yang
from typing import List
from rewards.get_reward_custom import get_reward


class UserReward():
    def __init__(
            self,
            agent_count,
            observation: List[dict] = None,
            **kwargs
    ):
        self.carbon_emission = None
        self.electricity_price = None
        self.agent_count = agent_count
        self.electricity_price_without_storage = None
        self.carbon_emission_without_storage = None
        self.ramping_delta = None
        self.load_factor_delta = None
        self.kwargs = kwargs

    def calculate(
            self
    ) -> List[float]:
        """CityLearn Challenge reward calculation.

        Notes
        -----
        This function is called internally in the environment's :meth:`citylearn.CityLearnEnv.step` function.
        """
        return get_reward(
            emission=self.carbon_emission,
            price=self.electricity_price,
            agent_count=self.agent_count,
            price_without_storage=self.electricity_price_without_storage,
            emission_without_storage=self.carbon_emission_without_storage,
            ramping_delta=self.ramping_delta,
            load_factor_delta=self.load_factor_delta
        )
