# -*- coding: utf-8 -*-
# @Time    : 2022/8/16 5:56 PM
# @Author  : Zhihu Yang, Abilmansur Zhumabekov
import random
import numpy as np


class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # for beta calculation
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.

        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        # state = np.expand_dims(state, 0)
        # next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0  # gives max priority if buffer is not empty else 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?)
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # calc P = p^a/sum(p^a)
        probs = prios ** self.alpha
        P = probs / probs.sum()

        # gets the indices depending on the probability p
        indices = np.random.choice(N, batch_size, p=P)
        if self.__len__() - 1 in indices:
            index = np.where(indices == self.__len__() - 1)
            indices = np.delete(indices, index)

        samples = [self.buffer[idx] for idx in indices]
        next_samples = [self.buffer[idx + 1] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Compute importance-sampling weight
        weights = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        _, _, _, next_next_states, _ = zip(*next_samples)
        return states, actions, rewards, next_states, dones, indices, weights, next_next_states

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio)

    def __len__(self):
        return len(self.buffer)


class HistoryBuffer(object):
    def __init__(
            self,
            history_len,
            num_features=4
    ):
        self.history_len = history_len
        self.num_features = num_features
        self.index = {
            hour + 1: 0 for hour in range(24)
        }
        self.buffer = {
            hour + 1: np.zeros((self.history_len, self.num_features)) for hour in range(24)
        }

    def add(
            self,
            hour,
            value
    ):
        self.buffer[hour][self.index[hour] % self.history_len] = value
        self.index[hour] += 1

    def get_past_mean(
            self,
            hour
    ):
        if self.index[hour] == 0:
            return np.zeros(self.num_features)
        elif self.index[hour] < self.history_len:
            return np.mean(self.buffer[hour][:self.index[hour]], 0)
        else:
            return np.mean(self.buffer[hour], 0)
