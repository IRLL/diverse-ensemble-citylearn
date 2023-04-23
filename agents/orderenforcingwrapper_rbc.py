import numpy as np
from gym.spaces import Box
import wandb
from agents.bin.buffer import HistoryBuffer
from agents.sac_dvd_agent import SAC_DvD_Agent
from agents.observation_space import observation_space_pre_dicts
from agents.rbc_agent_custom import SimpleRBCAgent
from agents.rbc_merlin import TOUPeakReductionAgent


def dict_to_action_space(aspace_dict):
    return Box(
        low=aspace_dict["low"],
        high=aspace_dict["high"],
        dtype=aspace_dict["dtype"],
    )


def action_space_to_dict(
        aspace
):
    """ Only for box space """
    return {"high": aspace.high,
            "low": aspace.low,
            "shape": aspace.shape,
            "dtype": str(aspace.dtype)
            }


def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations}
    return obs_dict



class OrderEnforcingAgent:
    def __init__(self, rbc_type, train_id=1):
        self.time_step = 0
        self.time_steps = 8760
        self.rbc_type = rbc_type
        assert rbc_type in ["xgboost", "merlin"], f"Unsupported rbc_type: {rbc_type}"
        if rbc_type == "xgboost":
            self.rbc_agent = SimpleRBCAgent(train_id=train_id)
        else:
            self.rbc_agent = TOUPeakReductionAgent()
        self.observation_space = None
        self.action_space = None
        self.num_buildings = None

    def register_reset(self, obs_dict):
        """ for each policy in the population reset its corresponding environment, obtain observations and return actss
        envs: list of env's, one env per policy
        is_train: whether training or not
        return actss: list of policy_actions's, where one policy_actions holds actions of the policy for all buildings
        """
        self.observation_space = obs_dict["observation_space"][0]
        self.action_space = obs_dict["action_space"][0]
        self.num_buildings = len(obs_dict["action_space"])
        # obss_raw = []
        # for policy_id, env in enumerate(envs):
        #     obs_raw = env.reset()
        #     obss_raw.append(obs_raw)
        #     if policy_id == 0:
        #         self.action_space = env.action_space[0]
        #         self.observation_space = observation_space_pre_dicts
        #         self.num_buildings = len(obs_raw)

        self.time_step = 0
        self.rbc_agent.reset()

        observations = obs_dict["observation"]
        actions = []
        avg_pv_gen = np.mean([obs[21] for obs in observations])
        for ag_i, obs in enumerate(observations):
            if self.rbc_type == "xgboost":
                act = self.rbc_agent.compute_action(obs, avg_pv_gen, ag_i, self.time_step)
                actions.append(act)
            else:
                actions.append(self.rbc_agent.compute_action(obs))

        self.time_step = 1
        return actions

    def compute_action(self, observations):
        """

        :param obss_raw: list of policy_observations's, where one policy_observations holds observations for all buildings for one policy/environment
        :param done:
        :param rewards: list of policy_rewards's, where each policy_rewards holds rewards for all buildings for one policy/environment
        :param is_train: whether training or not
        :return actss: list of policy_actions's, where one policy_actions holds actions of the policy for all buildings
        """
        actions = []
        avg_pv_gen = np.mean([obs[21] for obs in observations])
        for ag_i, obs in enumerate(observations):
            if self.rbc_type == "xgboost":
                act = self.rbc_agent.compute_action(obs, avg_pv_gen, ag_i, self.time_step)
                actions.append(act)
            else:
                actions.append(self.rbc_agent.compute_action(obs))

        self.time_step += 1
        return actions

