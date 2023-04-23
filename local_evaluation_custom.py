import os
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

from agents.orderenforcingwrapper_rbc import OrderEnforcingAgent
from citylearn.citylearn import CityLearnEnv
from custom_utils.env_utils import CityLearnEnvCustomReward

exp_save = True  # if True, save experience: observations, actions, rewards

custom_reward = True
reward_str = "custom" if custom_reward else "default"
assert custom_reward is True, "only custom reward is supported"


class Constants:
    episodes = 1
    train_id = 1
    rbc_type = "merlin"
    # rbc_type = "xgboost"
    schema_path_custom_reward = f'./data/citylearn_challenge_2022_phase_all/schema_custom_reward_validation7months_{train_id}.json'
    # schema_path_custom_reward = './data/citylearn_challenge_2022_phase_all/schema_custom_reward_train5months_1.json'
    # schema_path_custom_reward = './data/citylearn_challenge_2022_phase_all/schema_custom_reward_first5months_allHouses.json'
    # schema_path_custom_reward = './data/citylearn_challenge_2022_phase_all/schema_custom_reward_allHouses.json'
    # obsact_save_path = f'./data/observations_ToU-RBC_allHouses.pkl'
    obsact_save_path = f'./data/observations_ToU-RBC_train{train_id}.pkl'
    # obsact_save_path = f'./data/observations_SimpleRBC_train{train_id}.pkl'

assert Constants.episodes == 1, "only one episode is supported"


def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
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
                "observation": observations }
    return obs_dict


def evaluate():
    print("Starting local evaluation")
    if custom_reward:
        env = CityLearnEnvCustomReward(schema=Constants.schema_path_custom_reward)
    else:
        # env = CityLearnEnv(schema=Constants.schema_path_default_reward)
        assert False, "only custom_reward=True is supported"
    agent = OrderEnforcingAgent(train_id=Constants.train_id, rbc_type=Constants.rbc_type)
    num_buildings = len(env.buildings)
    print(f"num_buildings: {num_buildings}\t env.time_steps: {env.time_steps}")

    obs_dict = env_reset(env)
    observations = obs_dict['observation']
    step_start = time.perf_counter()
    actions = agent.register_reset(obs_dict)
    agent_time_elapsed = 0
    agent_time_elapsed += time.perf_counter() - step_start
    if exp_save:
        num_features = obs_dict["observation_space"][0]["shape"][0]
        all_exps = np.zeros(shape=(num_buildings, num_features+2, Constants.episodes*(env.time_steps)))  # (num buildings, obs dim + 1, T)
        # num_features for observation, 1 for action, 1 for reward; in total num_features+2 values are saved per step
        all_exps[:, :num_features, 0] = np.array(observations)
        all_exps[:, -2, 0] = np.array(actions).reshape(-1)


    episodes_completed = 0
    env_steps = 0  # env was reset, first observation obtained, first action computed, but env hasn't been stepped yet
    interrupted = False
    episode_metrics = []
    # returns = [0.0 for _ in range(len(obs_dict['building_info']))]

    try:
        while True:
            observations, rewards, done, _ = env.step(actions)
            env_steps += 1
            if done:
                episodes_completed += 1
                price_costs, emission_costs, total_price_cost, total_emission_cost, grid_cost = env.evaluate()
                metrics = {
                    "avg_building_cost": round(0.5 * (total_price_cost + total_emission_cost), 5),
                    "building_costs": {b_i: round(0.5 * (p + e), 5)
                                       for b_i, (p, e) in enumerate(zip(price_costs, emission_costs))},
                    "avg_price_cost": round(total_price_cost, 5),
                    "avg_emission_cost": round(total_emission_cost, 5),
                    "price_costs": {b_i: round(p, 5) for b_i, p in enumerate(price_costs)},
                    "emission_costs": {b_i: round(e, 5) for b_i, e in enumerate(emission_costs)},
                    "grid_cost": round(grid_cost, 5),

                }
                print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )

                obs_dict = env_reset(env)

                step_start = time.perf_counter()
                actions = agent.register_reset(obs_dict)
                agent_time_elapsed += time.perf_counter()- step_start
            else:
                step_start = time.perf_counter()
                actions = agent.compute_action(observations)
                agent_time_elapsed += time.perf_counter()- step_start

            if exp_save:
                all_exps[:, :num_features, env_steps] = np.array(observations)
                all_exps[:, -2, env_steps] = np.array(actions).reshape(-1)
                all_exps[:, -1, env_steps-1] = rewards['emission']+rewards['price']

            if env_steps % 1000 == 0:
                print(f"Env Steps: {env_steps}, Num episodes: {episodes_completed}")

            if episodes_completed >= Constants.episodes:
                break

    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True
    
    if not interrupted:
        print("=========================Completed=========================")

    print(f"Total time taken by agent: {agent_time_elapsed}s")

    # save experiences
    if exp_save:
        with open(Constants.obsact_save_path, 'wb') as fout:
            pickle.dump(all_exps, fout, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    evaluate()
