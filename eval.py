import numpy as np
import os
import argparse
import wandb
from agents.orderenforcingwrapper import OrderEnforcingAgent
from custom_utils.env_utils import CityLearnEnvCustomReward


parser = argparse.ArgumentParser()

parser.add_argument("--project-name", type=str, default="ALA_evaluation_std_ddof0")
# parser.add_argument("--label", type=str, default="")
parser.add_argument("-p", "--population-size", type=int)
parser.add_argument("-m", "--model-path", type=str, default=None)
parser.add_argument("-t", "--train-id", type=int, default=1)
parser.add_argument("-i", "--trial-id", type=int, default=1)
parser.add_argument("-r", "--rbc-weight", type=float, default=0.0)
parser.add_argument("--ensemble-type", type=str, default="std-weighted")
parser.add_argument("--eval-episodes", default=1, type=int)
parser.add_argument("--episodes-trained", default=80, type=int)

parser.add_argument("--array-id", default=0, type=int)
parser.add_argument("--paths-file", default=None, type=str)

args = parser.parse_args()
args.schema_path = f"./data/citylearn_challenge_2022_phase_all/schema_custom_reward_validation7months_{args.train_id}.json"
assert args.array_id > 0
# if args.array_id > 0:
args.train_id = 1 + int((args.array_id-1) % 3)
args.trial_id = 1 + int((args.array_id-1) // 3)
with open(args.paths_file) as f:
    lines = f.readlines()
    args.model_path = lines[args.array_id-1].strip()
    args.label = args.model_path.split('/')[-2]
    args.model_path = os.path.join(args.model_path, f"episodes_{args.episodes_trained}")

# args.label = f"p{args.p}-dvd{}"
run_name = f"p{args.population_size}-rbc{args.rbc_weight}-train{args.train_id}-{args.label}-trial{args.trial_id}"

wandb.config = {
    "population_size": args.population_size,
    "train_id": args.train_id,
    "trial_id": args.trial_id,
    "rbc_weight": args.rbc_weight,
    "ensemble_type": args.ensemble_type,
    "label": args.label,
    "model_path": args.model_path,
    "episodes_trained": args.episodes_trained
}

wandb.init(project=args.project_name, entity="abilmansplus", name=run_name, config=wandb.config)


def evaluate():
    print("Starting local evaluation")
    print(f"wandb.config: {wandb.config}")

    env = CityLearnEnvCustomReward(schema=args.schema_path)
    agent = OrderEnforcingAgent(population_size=args.population_size, ensemble_type=args.ensemble_type,
                                rbc_act_weight=args.rbc_weight, train_id=args.train_id)

    # agent_time_elapsed = 0
    # step_start = time.perf_counter()
    actss, ensemble_logs = agent.register_reset(envs=[env], is_train=False, model_path=args.model_path)
    actions = actss[0]
    # agent_time_elapsed += time.perf_counter() - step_start

    episodes_completed = 0
    num_steps = 0

    action_deviation_history, sigma_variation_history = [], []
    if "average_action_deviation" in ensemble_logs:
        action_deviation_history.append(ensemble_logs["average_action_deviation"])
        sigma_variation_history.append(ensemble_logs["average_sigma_coef_of_variation"])

    while episodes_completed < args.eval_episodes:
        observations, _, done, _ = env.step(actions)
        num_steps += 1
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
            print(
                f"episodes: {episodes_completed} | steps: {num_steps} | metrics: {metrics}")
            log_dict = {"steps": num_steps, "episodes": episodes_completed}
            log_dict.update({f"scores_eval/{key}": val for key, val in metrics.items()})
            wandb.log(log_dict)
            # step_start = time.perf_counter()
            actss, ensemble_logs = agent.register_reset(envs=[env], is_train=False)
            actions = actss[0]
            # agent_time_elapsed += time.perf_counter() - step_start
        else:
            # step_start = time.perf_counter()
            actss, _, ensemble_logs = agent.compute_action([observations], is_train=False)
            actions = actss[0]
            # agent_time_elapsed += time.perf_counter() - step_start

        if "average_action_deviation" in ensemble_logs:
            action_deviation_history.append(ensemble_logs["average_action_deviation"])
            sigma_variation_history.append(ensemble_logs["average_sigma_coef_of_variation"])
        if num_steps % 100 == 0:
            log_dict = {"steps": num_steps}
            log_dict.update({
                "average_action_deviation/last100stepsAvg": np.mean(action_deviation_history[-100:]),
                "average_action_deviation/avgUptoStep": np.mean(action_deviation_history),
                "average_sigma_coef_of_variation/last100stepsAvg": np.mean(sigma_variation_history[-100:]),
                "average_sigma_coef_of_variation/avgUptoStep": np.mean(sigma_variation_history),

            })
            wandb.log(log_dict)
        if num_steps % 1000 == 0:
            print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")


    # print(f"Total time taken by agent: {agent_time_elapsed}s")


if __name__ == '__main__':
    evaluate()
