import numpy as np
import pickle
import joblib
from collections import defaultdict
import pandas as pd


solar_regression_coefs = (
    np.array([1.40808803, -0.60781433, 0.23914281, -0.15654575]),  # trained on first 5 months of data, on 5 phase I buildings
    np.array([1.45883576, -0.63275423,  0.1984367, -0.1418985]),  # trained on first 5 months of data, on 5 phase II buildings
    np.array([1.53724313, -0.6656645,  0.12775744, -0.11410136])  # trained on first 5 months of data, on 5 phase III buildings
)


def simple_rbc_policy(observation):
    hour = observation[2]
    non_shiftable_load = observation[20]
    solar_generation = observation[21]

    load_minus_pv = non_shiftable_load - solar_generation

    if 15 <= hour <= 19:
        actmin = -0.4
    else:
        actmin = -0.1
    if load_minus_pv < 0.0 and not (15 <= hour <= 19):
        action = min(-load_minus_pv / 6.4, 1.0)  # try putting all excess solar power into battery
    elif load_minus_pv > 0.0:
        action = max(-load_minus_pv / 6.4, actmin)
    else:
        action = 0.0

    action = np.array([action])
    assert -1.0 <= action <= 1.0
    return action


class SimpleRBCAgent:
    def __init__(self, train_id):
        """

        :param train_id: group id of buildings used to train its load and solar predictors
        """
        # self.action_space = {}
        self.fixed_load_history = defaultdict(list)
        self.fixed_load_history_hourly = {h: defaultdict(list) for h in range(1, 25)}
        self.pv_gen_history = defaultdict(list)
        self.fixed_load_predictor = joblib.load(f"data/rbc/load_predictor_xgboost_first5months_phase{train_id}.sav")
        self.solar_regression_coefs = solar_regression_coefs[train_id-1]
        # self.avg_pv_gen_prev = 0.0  # average solar generation across houses in previous time-step

    # def set_action_space(self, agent_id, action_space):
    #     self.action_space[agent_id] = action_space

    def reset(self):
        self.fixed_load_history = defaultdict(list)
        self.fixed_load_history_hourly = {h: defaultdict(list) for h in range(1, 25)}
        self.pv_gen_history = defaultdict(list)

    def next_step_predictor_policy(self, observation, avg_pv_gen: float, agent_id: int, env_steps: int):
        t_upcoming = env_steps + 1
        past_hour = observation[2]
        upcoming_hour = 1 if past_hour == 24.0 else past_hour + 1
        hour_sin = np.sin(upcoming_hour/24 * 2 * np.pi)
        hour_cos = np.cos(upcoming_hour/24 * 2 * np.pi)
        past_14day_avg_for_hour = np.mean(self.fixed_load_history_hourly[upcoming_hour][agent_id][-14:])
        load_history = [self.fixed_load_history[agent_id][t] for t in reversed(range(t_upcoming-24, t_upcoming))]  # lag_1,2,...,24
        features = [hour_sin, hour_cos, past_14day_avg_for_hour]
        features += load_history

        feature_names = ['hour_sin', 'hour_cos', 'load_moving_avg_for_hour', 'non_shiftable_load_lag1',
                         'non_shiftable_load_lag2', 'non_shiftable_load_lag3',
                         'non_shiftable_load_lag4', 'non_shiftable_load_lag5',
                         'non_shiftable_load_lag6', 'non_shiftable_load_lag7',
                         'non_shiftable_load_lag8', 'non_shiftable_load_lag9',
                         'non_shiftable_load_lag10', 'non_shiftable_load_lag11',
                         'non_shiftable_load_lag12', 'non_shiftable_load_lag13',
                         'non_shiftable_load_lag14', 'non_shiftable_load_lag15',
                         'non_shiftable_load_lag16', 'non_shiftable_load_lag17',
                         'non_shiftable_load_lag18', 'non_shiftable_load_lag19',
                         'non_shiftable_load_lag20', 'non_shiftable_load_lag21',
                         'non_shiftable_load_lag22', 'non_shiftable_load_lag23',
                         'non_shiftable_load_lag24']
        assert len(features) == len(feature_names), f"expected len(features)={len(feature_names)}"


        df = pd.DataFrame([features], columns=feature_names)
        predicted_load = self.fixed_load_predictor.predict(df)
        predicted_load = predicted_load[0]

        # predict solar
        pv_gen_history = [self.pv_gen_history[agent_id][t] for t in [env_steps, env_steps-1]]
        avg_pv_gen_prev = np.mean([self.pv_gen_history[j][env_steps-1] for j in self.pv_gen_history.keys()])
        avg_pv_history = [avg_pv_gen, avg_pv_gen_prev]
        pv_features = np.array(pv_gen_history + avg_pv_history)
        pv_prediction = sum(pv_features*self.solar_regression_coefs)
        pv_prediction = max(pv_prediction, 0.0)
        load_minus_pv_pred = predicted_load - pv_prediction
        past_hour = observation[2]

        # avg score on train: 0.8257
        if 15 <= past_hour <= 19:
            actmin = -0.4  # ~ -0.5*5/6.4
            actmax = 0.2 * 5 / 6.4
        else:
            actmin = -0.1  # ~ -0.13*5/6.4
            actmax = 0.3 * 5 / 6.4

        if load_minus_pv_pred < 0.0:
            action = min(-load_minus_pv_pred / 6.4, actmax)  # try putting all excess solar power into battery
        elif load_minus_pv_pred > 0.0:
            action = max(-load_minus_pv_pred / 6.4, actmin)
        else:
            action = 0.0

        action = np.array([action])
        assert -1 <= action <= 1
        return action

    def compute_action(self, observation, avg_pv_gen, agent_id, env_steps):
        """Get observation return action"""
        self.fixed_load_history[agent_id].append(observation[20])
        self.pv_gen_history[agent_id].append(observation[21])
        hour = observation[2]
        self.fixed_load_history_hourly[hour][agent_id].append(observation[20])
        # train scores:
        # rbc + prediction: indiv 0.7802, overall 0.8257
        # simple_rbc_policy: indiv 0.8033, overall 0.8678
        if 23 <= env_steps < 8759:
            action = self.next_step_predictor_policy(observation=observation, avg_pv_gen=avg_pv_gen, agent_id=agent_id, env_steps=env_steps)
        else:
            action = simple_rbc_policy(observation)
        # action = simple_rbc_policy(observation)

        return action  # e.g.: array([0.091], dtype=float32)
