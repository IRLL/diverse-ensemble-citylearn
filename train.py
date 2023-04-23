import numpy as np
import time
from datetime import datetime
import os
import argparse

import wandb

from agents.orderenforcingwrapper import OrderEnforcingAgent
from custom_utils.env_utils import CityLearnEnvCustomReward
from rewards.get_reward_custom import reward_descriptor


parser = argparse.ArgumentParser()

parser.add_argument('--wandb-entity', type=str, default="", help="'entity' argument passed to wandb.init(), name of a wandb team or user")
parser.add_argument('--project-name', type=str, default="diverse-ensemble-citylearn", help="wandb project name")
parser.add_argument('--array-id', type=int, default=0, help="job array id, helpful when submitting an array of jobs;"
                                                            "to run 15 trials, submit a job with array-id=1-15;"
                                                            "see 'if args.array_id >= 1' block in this code for details")

parser.add_argument('--run-label', type=str, default="", help="used to identify checkpoints and logs")
parser.add_argument('-p', '--population-size', type=int, default=1, help="number of actors trained with policy diversity")
parser.add_argument('--dvd-coef', type=float, default=0.0, help="DvD loss coefficient")
parser.add_argument('--ensemble-type', type=str, default="std-weighted", help="kind of ensemble to use:"
                                                                              "'std-weighted' - give 1/std weight to each action"
                                                                              "'min-std' - pick action with smallest std")

parser.add_argument('--rbc-weight', type=float, default=0.0, help="rbc can be used during evaluation"
                                                                  "0.0 - don't use rbc"
                                                                  "0.0<x<=1.0 - action = x*rbc_action + (1-x)*drl_ensemble_action")

parser.add_argument('--episodes', type=int, default=120)
parser.add_argument('-e', '--eval-every-episodes', type=int, default=1)
parser.add_argument('--train-id', type=int, default=1, help="data partition id used for training, rest are used for validation")
parser.add_argument('--train-months', type=int, default=5, help="number of initial months to be used for training")
parser.add_argument('--explore-episodes', type=int, default=8)
args = parser.parse_args()

if args.array_id >= 1:  # doing hyperparameter sweep on cluster
    # value_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # dvd_coef
    # value_list = [0]
    value_list = [1, 2, 3]  # train_id
    # value_list = [1, 2, 4, 8]
    num_trials = 5
    value_list = value_list * num_trials
    # args.dvd_coef = value_list[args.array_id-1]
    args.train_id = value_list[args.array_id-1]
    # args.population_size = value_list[args.array_id-1]
    # args.explore_episodes = value_list[args.array_id-1]

args.schema_path_train = f'./data/citylearn_challenge_2022_phase_all/' \
                         f'schema_custom_reward_train{args.train_months}months_{args.train_id}.json'
args.schema_path_eval = f'./data/citylearn_challenge_2022_phase_all/' \
                        f'schema_custom_reward_validation{12-args.train_months}months_{args.train_id}.json'
# args.schema_path_eval = args.schema_path_train

start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
args.model_path = os.getcwd() + f'/agents/model_weights/train/{args.run_label}/job{args.array_id}_{start_datetime}/'

wandb.config = {
    "label": args.run_label,
    "start_datetime": start_datetime,
    "array_id": args.array_id,
    "train-id": args.train_id,
    "eval_every_episodes": args.eval_every_episodes,
    "episodes": args.episodes,
    "reward": reward_descriptor,
    "dvd_coef": args.dvd_coef,
    "population_size": args.population_size,
    "ensemble-type": args.ensemble_type,
    "rbc-act-weight": args.rbc_weight,
    "train-months": args.train_months,
    "explore-episodes": args.explore_episodes
}

run_name = f"{args.run_label}_job{args.array_id}_{start_datetime}"
if args.wandb_entity != "":
    wandb.init(
        project=args.project_name, name=run_name, entity=args.wandb_entity, config=wandb.config,
        settings=wandb.Settings(start_method='fork')  # delete the 'start_method' argument if running from Windows desktop
    )


def save_model(
        agent,
        episodes
):
    path = args.model_path + 'episodes_{}/'.format(episodes)
    agent.save_model(path)


def train():
    print("Starting local train")
    train_envs = [CityLearnEnvCustomReward(schema=args.schema_path_train) for _ in range(args.population_size)]
    eval_envs = [CityLearnEnvCustomReward(schema=args.schema_path_eval)]  # only one environment

    exploration_steps = int(args.explore_episodes/args.population_size)*args.train_months*30*24
    print(f"exploration_steps: {exploration_steps}")
    wandb.config.update({"exploration_steps": exploration_steps})
    agent = OrderEnforcingAgent(diversity_importance=args.dvd_coef, population_size=args.population_size,
                                ensemble_type=args.ensemble_type, rbc_act_weight=args.rbc_weight,
                                train_after_steps=exploration_steps, train_id=args.train_id, seed=args.array_id)

    training = True
    envs = train_envs if training else eval_envs

    actss, _ = agent.register_reset(envs=envs, is_train=training)

    train_episodes_completed = 0
    num_train_steps = 0

    train_episode_logs = {"loss_critic": [], "loss_actors_per_actor": [], "loss_dvd": [], "dvd_kernel_determinant": []}
    train_episode_reward_sums = [{"price": 0.0, "emission": 0.0} for _ in range(args.population_size)]  # \sum_t (1/N \sum_b reward_{t, b}), where t is a timestep, b - building, N - number of buildings in an environment
    eval_episode_reward_sum = {"price": 0.0, "emission": 0.0}
    lowest_eval_cost = np.inf
    while train_episodes_completed < args.episodes or not training:
        obss, rewards, dones = [], [], []
        for env, actions in zip(envs, actss):
            observations, rew, done, _ = env.step(actions)
            obss.append(observations)
            rewards.append(rew)
            dones.append(done)

        if training:
            num_train_steps += args.population_size
            for pol_id in range(args.population_size):
                # accumulate average reward across buildings, until the end of the episode
                for key in rewards[0].keys():
                    train_episode_reward_sums[pol_id][key] += np.mean(rewards[pol_id][key])
        else:
            # accumulate average reward across buildings, until the end of the episode
            for key in rewards[0].keys():
                eval_episode_reward_sum[key] += np.mean(rewards[0][key])

        actss, update_logs, _ = agent.compute_action(obss, dones[0], rewards, is_train=training)
        if training and update_logs is not None:
            for key in train_episode_logs.keys():
                train_episode_logs[key].append(update_logs[key])

        if any(dones):
            assert all(dones)
            if training:
                train_episodes_completed += args.population_size
            log_dict = {"train_episodes": train_episodes_completed, "population_size": args.population_size}
            if training and len(train_episode_logs["loss_critic"]) > 0:
                for key, values in train_episode_logs.items():
                    log_dict.update({f"train_episode_logs/{key}/min": np.min(values),
                                     f"train_episode_logs/{key}/median": np.median(values),
                                     f"train_episode_logs/{key}/mean": np.mean(values),
                                     f"train_episode_logs/{key}/max": np.max(values)})
                train_episode_logs = {"loss_critic": [], "loss_actors_per_actor": [], "loss_dvd": [], "dvd_kernel_determinant": []}
                log_dict.update({f"train_episode_reward_sum/policy_{i}": train_episode_reward_sums[i]
                                 for i in range(args.population_size)})
                train_episode_reward_sums = [{"price": 0.0, "emission": 0.0} for _ in range(args.population_size)]
            else:
                log_dict.update({"validation_episode_reward_sum": eval_episode_reward_sum})
                eval_episode_reward_sum = {"price": 0.0, "emission": 0.0}

            for policy_id, env in enumerate(envs):
                price_costs, emission_costs, total_price_cost, total_emission_cost, grid_cost = env.evaluate()
                metrics = {
                    "avg_building_cost": round(0.5*(total_price_cost+total_emission_cost), 5),
                    "building_costs": {b_i: round(0.5 * (p + e), 5)
                                       for b_i, (p, e) in enumerate(zip(price_costs, emission_costs))},
                    "avg_price_cost": round(total_price_cost, 5),
                    "avg_emission_cost": round(total_emission_cost, 5),
                    "price_costs": {b_i: round(p, 5) for b_i, p in enumerate(price_costs)},
                    "emission_costs": {b_i: round(e, 5) for b_i, e in enumerate(emission_costs)},
                    "grid_cost": round(grid_cost, 5),

                }
                # if np.any(np.isnan(metrics)):
                #     raise ValueError("Episode metrics are nan")
                if not training:
                    policy_id = "eval"
                    if metrics["avg_building_cost"] < lowest_eval_cost:
                        lowest_eval_cost = metrics["avg_building_cost"]
                    log_dict.update({f"scores_validation/avg_building_cost_lowest": lowest_eval_cost})
                print(f"policy_{policy_id} | train episodes: {train_episodes_completed} | train steps: {num_train_steps} | metrics: {metrics}")

                env_label = "_train" if training else "_validation"
                log_dict.update({f"scores{env_label}/{key}": val for key, val in metrics.items()})

            if args.wandb_entity != "":
                wandb.log(log_dict)

            if not training and train_episodes_completed > 0 and train_episodes_completed % 10 == 0:
                save_model(agent=agent, episodes=train_episodes_completed)
                print("Model saved")

            if training and train_episodes_completed >= args.explore_episodes and \
                    (train_episodes_completed % args.eval_every_episodes == 0 or train_episodes_completed == args.explore_episodes):
                training = False
            else:
                training = True
            envs = train_envs if training else eval_envs
            actss, _ = agent.register_reset(envs, is_train=training)


if __name__ == '__main__':
    train()
