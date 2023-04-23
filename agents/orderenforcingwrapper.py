import numpy as np
from gym.spaces import Box
import wandb
from agents.bin.buffer import HistoryBuffer
from agents.sac_dvd_agent import SAC_DvD_Agent
from agents.observation_space import observation_space_pre_dicts
from agents.rbc_agent_custom import SimpleRBCAgent


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
    def __init__(self, diversity_importance=None, population_size=None, ensemble_type=None, rbc_act_weight=None,
                 train_after_steps=8759, train_id=1, seed=0):
        self.time_step = 0
        self.time_steps = 8760
        self.agent = SAC_DvD_Agent(diversity_importance=diversity_importance, population_size=population_size,
                                   train_after_steps=train_after_steps, seed=seed)
        self.rbc_agent = SimpleRBCAgent(train_id=train_id)
        self.population_size = population_size  # number of diverse actors/policies to train
        self.histories = None
        self.history_size = 14
        self.observation_space = None
        self.action_space = None
        self.num_buildings = None
        self.last_observations_features = None
        self.use_ensemble = None
        self.ensemble_type = ensemble_type
        self.rbc_act_weight = rbc_act_weight
        # if rbc_act_weight is "ensemble-disagreement-based", following is used:
        self.rbc_base_weight = 0.4
        # self.running_ensemble_std_discount = 0.97 #0.99
        self.ensemble_std_cumulative = 0.0

    def get_ensemble_actss(self, actss, obss_raw):
        actss_ensemble = [[]]
        ensemble_logs = {}
        act_deviations_by_building = []
        sigma_coef_of_variation_by_building = []
        for b_i in range(self.num_buildings):
            ensemble_act_means = [actss[pol_i][b_i][0] for pol_i in range(self.population_size)]
            ensemble_act_stds = [actss[pol_i][b_i][1] for pol_i in range(self.population_size)]

            if self.ensemble_type == "std-weighted":
                weights = [1 / (std + 1e-6) for std in ensemble_act_stds]
            elif self.ensemble_type == "simple-averaging":
                weights = [1.0 for _ in ensemble_act_stds]
            elif self.ensemble_type == "min-std":
                weights = [1 if std == min(ensemble_act_stds) else 0 for std in ensemble_act_stds]
            else:
                assert False, "Unrecognized ensemble_type"
            assert sum(weights) > 0

            drl_action = sum([w * a for w, a in zip(weights, ensemble_act_means)]) / sum(weights)
            if self.rbc_act_weight > 0:
                avg_pv_gen = np.mean([obss_raw[0][j][21] for j in range(self.num_buildings)])
                rbc_action = self.rbc_agent.compute_action(observation=obss_raw[0][b_i], avg_pv_gen=avg_pv_gen,
                                                           agent_id=b_i, env_steps=self.time_step)
                action = self.rbc_act_weight * rbc_action + (1-self.rbc_act_weight)*drl_action
                all_actions = ensemble_act_means + [rbc_action]
            else:
                assert self.rbc_act_weight == 0.0
                action = drl_action
                all_actions = ensemble_act_means
            actss_ensemble[0].append(action)

            # compute discrepancies of actions
            # act_avg = np.mean(all_actions)
            # act_deviations = [np.abs(a - act_avg) for a in all_actions]
            # act_deviations_by_building.append(np.mean(act_deviations))
            act_deviations_by_building.append(np.std(all_actions, ddof=0))
            # compute variation of stds
            sigma_coef_of_variation = np.std(ensemble_act_stds, ddof=0)/np.mean(ensemble_act_stds)
            sigma_coef_of_variation_by_building.append(sigma_coef_of_variation)

            # if len(actss) == 2:
            #     action_abs_deltas.append(np.abs(ensemble_act_means[0]-ensemble_act_means[1]))
            # elif len(actss) == 1 and self.rbc_act_weight > 0:
            #     action_abs_deltas.append(np.abs(drl_action - rbc_action))

        # if len(action_abs_deltas) > 0:
        #     ensemble_logs.update({"action_abs_delta_per_house": np.mean(action_abs_deltas)})
        ensemble_logs.update({"average_action_deviation": np.mean(act_deviations_by_building),
                              "average_sigma_coef_of_variation": np.mean(sigma_coef_of_variation_by_building)})
        return actss_ensemble, ensemble_logs

    def register_reset(self, envs, is_train, model_path=None):
        """ for each policy in the population reset its corresponding environment, obtain observations and return actss
        envs: list of env's, one env per policy
        is_train: whether training or not
        return actss: list of policy_actions's, where one policy_actions holds actions of the policy for all buildings
        """
        self.use_ensemble = not is_train
        obss_raw = []
        for policy_id, env in enumerate(envs):
            obs_raw = env.reset()
            obss_raw.append(obs_raw)
            if policy_id == 0:
                self.action_space = env.action_space[0]
                self.observation_space = observation_space_pre_dicts
                self.num_buildings = len(obs_raw)

        if self.use_ensemble:
            # make copies of the observation (all policies will get the same observations)
            for _ in range(self.population_size-1):
                obss_raw.append(obss_raw[0])
            self.rbc_agent.reset()

        self.histories = [{}]*self.population_size
        self.last_observations_features = [{}]*self.population_size
        obss = []
        for policy_id in range(self.population_size):
            for agent_id in range(self.num_buildings):
                self.histories[policy_id][agent_id] = HistoryBuffer(self.history_size, num_features=2)
                self.last_observations_features[policy_id][agent_id] = [0.070, 0, 0, 0.21, 0.21, 0.21, 0.21]
            obs = self.get_obs(obss_raw[policy_id], policy_id=policy_id)
            obss.append(obs)

        actss = self.agent.register_reset(obss=obss, action_space=self.action_space,
                                          observation_space=self.observation_space, is_train=is_train,
                                          model_path=model_path)
        assert self.num_buildings == self.agent.num_agents

        self.time_step = 0

        ensemble_logs = None
        if self.use_ensemble:
            # actss: [policy1_acts, policy2_acts, ...]; policy1_acts: [(b1_act_mean, b1_act_std), (b2_act_mean, b2_act_std), ...]

            actss, ensemble_logs = self.get_ensemble_actss(actss=actss, obss_raw=obss_raw)

        self.time_step = 1
        return actss, ensemble_logs

    def compute_action(self, obss_raw, done=False, rewards=None, is_train=True):
        """

        :param obss_raw: list of policy_observations's, where one policy_observations holds observations for all buildings for one policy/environment
        :param done:
        :param rewards: list of policy_rewards's, where each policy_rewards holds rewards for all buildings for one policy/environment
        :param is_train: whether training or not
        :return actss: list of policy_actions's, where one policy_actions holds actions of the policy for all buildings
        """
        assert self.num_buildings is not None

        # these 2 lines are optional, results should be very similar without them
        if is_train and self.judge_done(obss_raw[0][0]):
            done = True

        if self.use_ensemble:
            # make copies of the observation (all policies will get the same observations)
            for _ in range(self.population_size-1):
                obss_raw.append(obss_raw[0])

        obss = []
        for policy_id in range(self.population_size):
            obs = self.get_obs(obss_raw[policy_id], policy_id=policy_id)
            obss.append(obs)

        actss, update_logs = self.agent.compute_action(obss=obss, is_train=is_train, rewards=rewards, done=done)
        ensemble_logs = None
        if self.use_ensemble:
            # actss: [policy1_acts, policy2_acts, ...]; policy1_acts: [[(b1_act_mean, b1_act_std)], [(b2_act_mean, b2_act_std)], ...]
            actss, ensemble_logs = self.get_ensemble_actss(actss=actss, obss_raw=obss_raw)

        self.time_step += 1
        return actss, update_logs, ensemble_logs

    def save_model(
            self,
            path
    ):
        self.agent.save_model(path)

    @staticmethod
    def judge_done(
            observation
    ):
        if int(observation[1]) == 7 and int(observation[2]) == 24:
            return True
        else:
            return False

    def get_obs(
            self,
            observation,
            policy_id
    ):
        """

        :param observation: raw observation from environment
        :param policy_id: policy id (from population)
        :return observation: raw observation + extra features
        """
        observation = observation.copy()
        hour = observation[0][2]  # past hour
        next_hour = (hour + 1.0) if hour < 23.99 else 1.0
        next_next_hour = (next_hour + 1.0) if next_hour < 23.99 else 1.0
        Sixth_next_hour = 1.0 + (hour-1+6) % 24
        Twelth_next_hour = 1.0 + (hour-1+12) % 24
        assert 0.99 < hour < 24.01
        assert 0.99 < next_hour < 24.01
        assert 0.99 < next_next_hour < 24.01
        assert 0.99 < Sixth_next_hour < 24.01
        assert 0.99 < Twelth_next_hour < 24.01

        observation_use = np.array(observation)
        observation_copy = observation_use[:, 20:-4]
        assert observation_copy.shape[1] == 4

        for i in range(self.num_buildings):
            # average value[-8:-6] of the next several hours
            avg_for_next_hour = self.histories[policy_id][i].get_past_mean(next_hour).tolist()
            avg_for_next_next_hour = self.histories[policy_id][i].get_past_mean(next_next_hour).tolist()
            avg_for_6th_next_hour = self.histories[policy_id][i].get_past_mean(Sixth_next_hour).tolist()
            avg_for_12th_next_hour = self.histories[policy_id][i].get_past_mean(Twelth_next_hour).tolist()

            avg_for_next_hour.extend(avg_for_next_next_hour)
            avg_for_next_hour.extend(avg_for_6th_next_hour)
            avg_for_next_hour.extend(avg_for_12th_next_hour)

            avg_for_next_hour.extend(self.last_observations_features[policy_id][i].copy())  # adds previous observations (but only carbon, fixed load, pv, prices)


            avg_for_next_hour.extend(observation[i])

            observation[i] = avg_for_next_hour
            self.histories[policy_id][i].add(hour, observation_copy[i, :2])

        left, right = observation_use[:, -9:-6], observation_use[:, -4:]
        combined = np.concatenate((left, right), axis=1)
        self.last_observations_features[policy_id] = combined.tolist()
        return observation
