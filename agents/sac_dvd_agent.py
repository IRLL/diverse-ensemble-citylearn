import os
import numpy as np
import copy

from agents.sac_agent import SAC_Agent
from agents.bin.sac_dvd import SAC_DvD


class SAC_DvD_Agent(SAC_Agent):
    def __init__(self, diversity_importance, population_size, train_after_steps, seed):
        super().__init__()
        self.population_size = population_size
        self.month = None
        self.day = None
        self.previous_actss = {}
        self.diversity_importance = diversity_importance
        self.train_after_steps = train_after_steps
        self.seed = seed

    def register_reset(
            self,
            obss,
            action_space,
            observation_space,
            is_train,
            model_path=None,
            **kwargs
    ):
        """

        :param obss: list of policy_observations's, where one policy_observations holds observations for all buildings for one policy/environment
        :param action_space: list of building action spaces
        :param observation_space: list of biulding observation spaces
        :param is_train: whether training or not
        :param model_path: directory that stores actor models and parameters
        :return actss: list of policy_actions's, where one policy_actions holds actions of the policy for all buildings
        """
        actss = []
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_agents = len(obss[0])
        # update month and day
        self.month = int(obss[0][0][-28])
        if self.month == 12.0:
            self.day = 28
        elif self.month == 7.0:
            self.day = 31
        elif self.month == 10.0:
            self.day = 29
        else:
            assert False, "start day is not specified"
        assert 1 <= self.month <= 12


        for policy_id, policy_observations in enumerate(obss):
            self.observation[policy_id] = {}
            policy_acts = []
            for agent_id, agent_observation in enumerate(policy_observations):
                agent_observation = self.get_states(observation=agent_observation)
                if self.agent is None:
                    self.agent = SAC_DvD(population_size=self.population_size, diversity_importance=self.diversity_importance,
                                         action_space=self.action_space, observation_space=self.observation_space,
                                         start_training_time_step=self.train_after_steps,
                                         end_exploration_time_step=self.train_after_steps, seed=self.seed)
                    if not is_train and model_path is not None:
                        # path = os.getcwd() + '/agents/model_weights/eval/'
                        self.agent.load_model(model_path)
                        # self.agent.deterministic_start_time_step = -1

                self.observation[policy_id][agent_id] = agent_observation
                update_time_step = is_train and (policy_id == self.population_size-1) and (agent_id == self.num_agents-1)
                act = self.agent.select_actions(agent_observation, is_train=is_train,
                                                last_agent=update_time_step,
                                                policy_id=policy_id)
                policy_acts.append(act)
            actss.append(policy_acts)

        self.previous_actss = copy.deepcopy(actss)
        return actss

    def compute_action(self, obss, is_train: bool = True, rewards=None, done=None, *args, **kwargs):

        """
        compute actions
        :param obss: list of policy_observations's, where one policy_observations holds observations for all buildings for one policy/environment
        :param rewards: list of policy_rewards's, where each policy_rewards holds rewards for all buildings for one policy/environment
        :param done:
        :param is_train: whether training or not
        :return actss: list of policy_actions's, where one policy_actions holds actions of the policy for all buildings
        """
        # update month and day
        if self.month == int(obss[0][0][-28]) and int(obss[0][0][-28+2]) == 1:
            self.day += 1
        else:
            self.month = int(obss[0][0][-28])
            self.day = 1
        assert 1 <= self.day <= 31 and 1 <= self.month <= 12

        actss = []
        for policy_id, policy_observations in enumerate(obss):
            policy_actions = []
            for agent_id, agent_observation in enumerate(policy_observations):
                if is_train:
                    agent_rewards = {rew_type: rewards[policy_id][rew_type][agent_id]
                                     for rew_type in rewards[policy_id].keys()}
                    losses = self.update_policy(
                        action=self.previous_actss[policy_id][agent_id],
                        reward=agent_rewards,
                        observation=obss[policy_id][agent_id],
                        done=done,
                        policy_id=policy_id,
                        agent_id=agent_id
                    )
                    agent_act = self.agent.select_actions(observations=self.observation[policy_id][agent_id], is_train=is_train,
                                                          last_agent=(policy_id == self.population_size-1) and (agent_id == self.num_agents-1),
                                                          policy_id=policy_id)
                else:
                    agent_observation = self.get_states(agent_observation)
                    agent_act = self.agent.select_actions(observations=agent_observation, is_train=is_train,
                                                          last_agent=False,
                                                          policy_id=policy_id)
                    losses = None
                policy_actions.append(agent_act)
            actss.append(policy_actions)

        self.previous_actss = copy.deepcopy(actss)
        return actss, losses

    def get_states(self, observation, *args, **kwargs):
        """
        Get states
        """
        if not self.encoder:
            self.encoder = self.get_encoder(self.observation_space)

        observation.insert(0, self.day)
        observation = np.hstack([e * o for o, e in zip(observation, self.encoder) if e is not None])
        self.observation_space['shape'] = (observation.size,)
        return observation

    def update_policy(
            self,
            action,
            reward,
            observation,
            done,
            policy_id,
            agent_id,
    ):
        """Update buffer and policy"""
        observation = self.get_states(observation)
        losses = self.agent.add_to_buffer(self.observation[policy_id][agent_id], action, reward, observation,
                                          last_agent=((policy_id == self.population_size-1) and (agent_id == self.num_agents-1)),
                                          done=done)
        self.observation[policy_id][agent_id] = observation
        return losses


    def save_model(self, path):
        self.agent.save_model(path)

    def load_model(self, path):
        self.agent.load_model(path)
