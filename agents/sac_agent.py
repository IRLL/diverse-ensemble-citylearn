# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 11:44 AM
# @Author  : Zhihu Yang, Abilmansur Zhumabekov
import os
import pickle
import numpy as np

from pandas.core.frame import DataFrame
from gym.spaces import Box
from agents.bin.sac import SAC
from agents.tool.normalization import periodic_normalization, normalize

drop_features = (  # dropping features with low mutual information score with both non_shiftable_load and solar_generation
    1,  # day_type
)

# 0, 1, 2,  # hour, day_type, month
# 3, 4, 5, 6,  # temperature and its predictions
# 7, 8, 9, 10,  # humidity and its predictions
# 11, 12, 13, 14,  # diffuse_solar_irradiance  (correlated with pv)
# 15, 16, 17, 18,  # direct_solar_irradiance, low correlation with solar_generation, but mutual information is ok
# 19,  # carbon_intensity  (important)
# 24, 25, 26, 27  # electricity_pricing (important)


def dict_to_action_space(aspace_dict):
    return Box(
        low=aspace_dict["low"],
        high=aspace_dict["high"],
        dtype=aspace_dict["dtype"],
    )


class SAC_Agent:

    def __init__(self):
        self.action_space = {}
        self.observation_space = {}
        self.agent = None
        self.observation = {}
        self.encoder = {}
        self.month = {}
        self.day = {}

        self.num_agents = None  # number of buildings, not the number of policies/actors

    def register_reset(
            self,
            observation,
            action_space,
            observation_space,
            agent_id,
            is_train,
    ):
        """Get the first observation after env.reset, return action"""
        self.action_space[agent_id] = action_space
        observation, observation_space = self.get_states(
            observation,
            observation_space,
            agent_id,
            True,
        )

        self.observation_space[agent_id] = observation_space

        if self.agent is None:
            self.agent = SAC(
                action_space=self.action_space[agent_id],
                observation_space=observation_space
            )
            if not is_train:
                path = os.getcwd() + '/agents/model_weights/eval/'
                self.agent.load_model(path)
                self.agent.deterministic_start_time_step = -1
        else:
            pass

        self.observation[agent_id] = observation
        return self.agent.select_actions(observation, is_train=is_train, last_agent=(agent_id == self.num_agents-1))

    def compute_action(
            self,
            observation,
            agent_id,
            is_train: bool = True,
            *args, **kwargs
    ):
        """Get observation return action"""

        if is_train:
            return self.agent.select_actions(self.observation[agent_id], is_train, last_agent=(agent_id == self.num_agents-1))
        else:
            observation, observation_space = self.get_states(
                observation,
                self.observation_space,
                agent_id,
                False,
            )
            return self.agent.select_actions(observation, is_train=is_train, last_agent=(agent_id == self.num_agents-1))

    def update_policy(
            self,
            action,
            reward,
            observation,
            done,
            agent_id
    ):
        """Update buffer and policy"""
        observation, observation_space = self.get_states(observation, self.observation_space, agent_id)
        self.agent.add_to_buffer(self.observation[agent_id], action, reward, observation, last_agent=(agent_id == self.num_agents-1), done=done)
        self.observation[agent_id] = observation

    def save_idx_model(
            self,
            path
    ):
        self.agent.save_model(path)

    def get_states(
            self,
            observation,
            observation_space,
            agent_id,
            reset: bool = False,
            *args, **kwargs
    ):
        """
        Get states
        """
        if not self.encoder:
            self.encoder = self.get_encoder(observation_space)

        if len(self.encoder) == 0:
            pass
        else:
            observation = self.add_feature(observation, agent_id, reset)  # one integer is added at the beginning
            # observation.insert(0, agent_id)
            observation = np.hstack([e * o for o, e in zip(observation, self.encoder) if e is not None])
            observation_space['shape'] = (observation.size,)
        return observation, observation_space

    def add_feature(
            self,
            observation,
            agent_id,
            reset
    ):
        if agent_id not in self.month or reset:
            self.month[agent_id] = observation[0]
            self.day[agent_id] = 31
        else:
            if self.month[agent_id] == int(observation[0]):
                self.day[agent_id] += 1
            else:
                self.month[agent_id] = observation[0]
                self.day[agent_id] = 1
        observation.insert(0, self.day[agent_id])

        return observation

    @staticmethod
    def get_encoder(
            observation_space,
    ):
        """
        Get encoder
        """
        obs_encode = []
        if observation_space['shape'] == (28,):
            index = 0
            assert(len(observation_space['high'])==len(observation_space['low']))
            original_obs_start_index = len(observation_space['high']) - 28
            for above, below in zip(observation_space['high'], observation_space['low']):
                if index - original_obs_start_index in drop_features:
                    obs_encode.append(None)
                elif index - original_obs_start_index in [0, 1, 2]:  # hour, daytype, month
                    obs_encode.append(periodic_normalization(above))
                else:
                    obs_encode.append(normalize(below, above))
                index += 1
            obs_encode.insert(0, periodic_normalization(31))  # converts 1 number to 2 via sin-cos transform
        return obs_encode
