import os
import json
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import List
from agents.bin.rlc import RLC
from agents.bin.buffer import PrioritizedReplay
from agents.bin.rl import Actor, Critic
from agents.bin.loss_dvd import DiversityLoss

charging_coef, discharging_coef = 0.9173, 1.0937  # estimates of average charging/discharging efficiencies


class SAC_DvD(RLC):
    def __init__(self, population_size, diversity_importance, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # internally defined
        self.alpha = 0.2  # entropy coefficient
        self.reward_scaling = 10.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.replay_buffer = PrioritizedReplay(int(self.replay_buffer_capacity))

        self.normalized = False
        self.obs_norm_mean = 0.0
        self.obs_norm_std = 1.0
        self.r_norm_mean = {"price": 0.0, "emission": 0.0}
        self.r_norm_std = {"price": 1.0, "emission": 1.0}

        self.critic = None
        self.critic_target = None
        self.critic_optimizer = None
        self.scheduler_critic = None

        self.num_obs_embedded = 20  # number of observations embedded for diversity loss
        self.diversity_importance = diversity_importance
        self.population = []
        self.population_size = population_size
        self.population_optimizer = None
        self.scheduler_population = None
        self.dvd_loss_function = DiversityLoss()

        self.set_networks()

    def set_networks(self):
        # init critic networks and optimizers
        self.critic = Critic(
            self.observation_dimension,
            self.action_dimension,
            self.hidden_dimension,
        ).to(self.device)
        self.critic_target = Critic(  # TODO: set to eval mode?
            self.observation_dimension,
            self.action_dimension,
            self.hidden_dimension,
        ).to(self.device)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.scheduler_critic = optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, 99 * 8759 * 2,
                                                                     eta_min=self.lr * 0.1)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # init population of actors and their optimizer
        population_params = []
        for _ in range(self.population_size):
            actor = Actor(
                self.observation_dimension,
                self.action_dimension,
                self.action_space,
                self.action_scaling_coefficient,
                self.hidden_dimension,
            ).to(self.device)
            population_params += list(actor.parameters())
            self.population.append(actor)

        self.population_optimizer = torch.optim.Adam(population_params, lr=self.lr)
        self.scheduler_population = optim.lr_scheduler.CosineAnnealingLR(self.population_optimizer, 99 * 8759 * 2, eta_min=self.lr * 0.1)


    def add_to_buffer(
            self,
            observations: List[float],
            actions: List[float],
            reward: dict,
            next_observations: List[float],
            last_agent: bool,
            done: bool = False,
    ):
        """

        :param observations:
        :param actions:
        :param reward:
        :param next_observations:
        :param last_agent: whether it is the last agent in an env step
        :param done:
        :return:
        """
        if self.normalized:
            observations = np.array(self.get_normalized_observations(observations), dtype=float)
            next_observations = np.array(self.get_normalized_observations(next_observations), dtype=float)
            reward = self.get_normalized_reward(reward)  # float
        else:
            reward = copy.deepcopy(reward)  # dict

        self.replay_buffer.push(observations, actions, reward, next_observations, done)

        if not last_agent or self.time_step < self.start_training_time_step or self.batch_size > len(self.replay_buffer):
            return None

        if not self.normalized:
            X = np.array([j[0] for j in self.replay_buffer.buffer], dtype=float)
            self.obs_norm_mean = np.nanmean(X, axis=0)
            self.obs_norm_std = np.nanstd(X, axis=0) + 1e-5
            for key in reward.keys():
                R = np.array([j[2][key] for j in self.replay_buffer.buffer], dtype=float)
                self.r_norm_mean[key] = np.nanmean(R, dtype=float)
                self.r_norm_std[key] = np.nanstd(R, dtype=float) / self.reward_scaling + 1e-5

            self.replay_buffer.buffer = [(
                np.hstack(
                    (np.array(self.get_normalized_observations(observations), dtype=float)).reshape(1, -1)[
                        0]),
                actions,
                self.get_normalized_reward(reward),
                np.hstack(
                    (np.array(self.get_normalized_observations(next_observations), dtype=float)).reshape(1,
                                                                                                           -1)[
                        0]),
                done
            ) for observations, actions, reward, next_observations, done in self.replay_buffer.buffer]
            self.normalized = True
            print(f"Calculated mean and std for observations and rewards.")

        k = 1 + len(self.replay_buffer) / self.replay_buffer_capacity
        batch_size = int(k * self.batch_size)
        update_times = int(k * self.update_per_time_step)

        tensor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
        losses = {"loss_critic": 0, "loss_actors_per_actor": 0, "loss_dvd": 0, "dvd_kernel_determinant": 0}
        for _ in range(update_times):
            obss, next_obss, actss, rewards, dones, indices = [], [], [], [], [], []
            for _ in range(self.population_size):
                observations, actions, reward, next_observations, done, idx, weights, _ = \
                    self.replay_buffer.sample(batch_size)

                observations = tensor(np.array(observations)).to(self.device)
                obss.append(observations)
                next_observations = tensor(np.array(next_observations)).to(self.device)
                next_obss.append(next_observations)
                actions = tensor(np.array(actions)).to(self.device)
                actss.append(actions)
                reward = tensor(np.array(reward)).unsqueeze(1).to(self.device)
                rewards.append(reward)
                done = tensor(np.array(done)).unsqueeze(1).to(self.device)
                dones.append(done)
                indices.append(idx)

            # Update Critic
            loss_critic = self.update_critic_and_memory(obss, actss, rewards, next_obss, dones, indices)

            # Update Actors
            # if self.time_step % self.actor_update_freq == 0:
            observations_for_embedding, _, _, _, _, _, _, _ = self.replay_buffer.sample(self.num_obs_embedded)
            # observations_for_embedding, _, _, _, _, _, _, _ = self.replay_buffer.sample(batch_size)
            observations_for_embedding = tensor(np.array(observations_for_embedding)).to(self.device)
            loss_actors, loss_dvd = self.update_actors(obss, observations_for_embedding)

            soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.tau
            )
            soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.tau
            )

            losses["loss_critic"] += loss_critic
            losses["loss_actors_per_actor"] += loss_actors/self.population_size
            losses["loss_dvd"] += loss_dvd
            losses["dvd_kernel_determinant"] += np.exp(-loss_dvd)
        return losses


    def update_critic_and_memory(
            self,
            obss,
            actss,
            rewards,
            next_obss,
            dones,
            indices
    ):
        all_obss = torch.cat(obss)
        all_actss = torch.cat(actss)
        all_rewards = torch.cat(rewards)
        all_next_obss = torch.cat(next_obss)
        all_not_dones = 1 - torch.cat(dones)
        all_indices = np.concatenate(indices)

        with torch.no_grad():
            next_actss, next_log_pis = [], []
            for actor, next_observations in zip(self.population, next_obss):
                new_next_actions, new_log_pi, _, _ = actor.sample(next_observations)
                next_actss.append(new_next_actions)
                next_log_pis.append(new_log_pi)
            all_next_actss = torch.cat(next_actss)
            all_next_log_pis = torch.cat(next_log_pis)
            target_Q1, target_Q2 = self.critic_target(all_next_obss, all_next_actss)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * all_next_log_pis
            target_Q = all_rewards + (all_not_dones * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(all_obss, all_actss)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()

        critic_loss.backward()
        self.critic_optimizer.step()
        self.scheduler_critic.step()

        # Update memory
        td_error1 = target_Q.detach() - current_Q1
        td_error2 = target_Q.detach() - current_Q2
        prios = abs(((td_error1 + td_error2) / 2.0 + 1e-5).squeeze())
        self.replay_buffer.update_priorities(all_indices, prios.data.cpu().numpy())
        return critic_loss.data.cpu().numpy()

    def compute_actor_loss(self, actor, obs):
        new_actions, log_pi, _, _ = actor.sample(obs)
        actor_Q1, actor_Q2 = self.critic(obs, new_actions)
        q_new_actions = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha * log_pi - q_new_actions).mean()
        return actor_loss

    def compute_diversity_loss(self, obs):
        embeddings = []
        for actor in self.population:
            _, _, act_mean, _ = actor.sample(obs)  # act_mean: tensor of shape (num observations embedded, 1)
            # obs: tensor of shape (num observations embedded, observation dim)
            act_mean = act_mean.flatten()  # tensor of shape (num observations embedded,)
            SoCs = obs[:, -6]*self.obs_norm_std[-6] + self.obs_norm_mean[-6]  # SoCs in range 0 to 1, tensor of shape (num observations embedded,)
            min_acts = torch.maximum(-SoCs/discharging_coef, -self.population[0].action_scale)
            max_acts = torch.minimum((1-SoCs)/charging_coef, self.population[0].action_scale)
            act_mean = torch.clamp(act_mean, min=min_acts, max=max_acts)  # remove inconsequential differences in actions
            embeddings.append(act_mean)
        embeddings = torch.stack(embeddings)  # tensor of shape (population size, embedding dim)
        assert len(embeddings.shape) == 2 and embeddings.shape[0] == self.population_size
        return self.dvd_loss_function(embeddings)

    def update_actors(self, obss, obs_for_embedding):
        loss_actors = 0
        for actor, obs in zip(self.population, obss):
            loss_actors += self.compute_actor_loss(actor, obs)
        loss_dvd = self.compute_diversity_loss(obs_for_embedding)
        total_actors_loss = loss_actors + self.diversity_importance * loss_dvd

        self.population_optimizer.zero_grad()
        total_actors_loss.backward()
        self.population_optimizer.step()
        self.scheduler_population.step()
        return loss_actors.data.cpu().numpy(), loss_dvd.data.cpu().numpy()

    def select_actions(self, observations: List[float], is_train: bool = True, last_agent: bool = False, policy_id=None):
        if is_train:
            if self.time_step <= self.end_exploration_time_step:
                actions = self.get_exploration_actions()
            else:
                actions = self.get_post_exploration_actions(observations, is_train=True, policy_id=policy_id)
        else:
            actions = self.get_post_exploration_actions(observations, is_train=False, policy_id=policy_id)

        if last_agent:
            self.next_time_step()

        return actions


    def get_post_exploration_actions(self, observations: List[float], is_train: bool, policy_id: int) -> List[float]:
        """Action sampling using policy, post-exploration time step"""
        with torch.no_grad():
            observations = np.array(self.get_normalized_observations(observations), dtype=float)
            observations = torch.FloatTensor(observations).unsqueeze(0).to(self.device)
            actions = self.population[policy_id].sample(observations)
            if is_train:
                actions = actions[2] if self.time_step >= self.deterministic_start_time_step else actions[0]
                actions = actions.detach().cpu().numpy()[0]
            else:
                act_mean = actions[2].detach().cpu().numpy()[0]
                act_std = actions[3].detach().cpu().numpy()[0]
                actions = (act_mean, act_std)

            return actions


    def get_exploration_actions(self) -> List[float]:
        return list(self.action_scaling_coefficient * self.action_space.sample())


    def get_normalized_reward(self, reward: dict) -> dict:
        normalized_rewards = {kind: (reward[kind] - self.r_norm_mean[kind]) / self.r_norm_std[kind]
                              for kind in reward.keys()}
        overall_reward = 1/2*(normalized_rewards['price']+normalized_rewards['emission'])
        return overall_reward

    def get_normalized_observations(self, observations: List[float]) -> List[float]:
        return ((np.array(observations, dtype=float) - self.obs_norm_mean) / self.obs_norm_std).tolist()

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        # torch.save(self.critic, path + 'critic.pt')
        # torch.save(self.critic_target, path + 'critic_target.pt')
        for actor_id, actor in enumerate(self.population):
            torch.save(actor, path + f'actor_{actor_id}.pt')
        parameter = {
            "obs_norm_mean": self.obs_norm_mean if self.obs_norm_mean is None else self.obs_norm_mean.tolist(),
            "obs_norm_std": self.obs_norm_std if self.obs_norm_std is None else self.obs_norm_std.tolist(),
            "r_norm_mean": self.r_norm_mean if self.r_norm_mean is None else self.r_norm_mean,
            "r_norm_std": self.r_norm_std if self.r_norm_std is None else self.r_norm_std,
        }
        with open(path + 'parameter.json', "w") as f:
            json.dump(parameter, f)

    def load_model(self, path):
        for actor_id in range(self.population_size):
            loaded_actor = torch.load(os.path.join(path, f'actor_{actor_id}.pt'), map_location=self.device)
            self.population[actor_id] = loaded_actor
        with open(os.path.join(path, 'parameter.json'), 'r') as f:
            parameter = json.load(f)
            self.normalized = True
            self.obs_norm_mean = np.array(parameter['obs_norm_mean'])
            self.obs_norm_std = np.array(parameter['obs_norm_std'])


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )
