import gym
import numpy as np

class ReplayMemory(object):
    def __init__(self, state_shape:tuple, action_space:int, capacity=1000000):
        self.states = np.zeros([capacity] + list(state_shape), dtype=float)
        self.actions = np.zeros((capacity,1), dtype=int)
        self.rewards = np.zeros((capacity, 1), dtype=float)
        self.q_values = np.zeros((capacity, action_space), dtype=float)
        self.done = np.zeros((capacity, 1), dtype=bool)
        self.estimate_error = np.zeros((capacity, 1), dtype=float)
        self.length = 0
        self.capacity = capacity
        self.discount_factor = 0.95
        self.err_threshold = 0.1

    def add(self, state, action, reward, done:bool, estimated_q:float):
        if self.is_full:
            raise Exception("replay memory is fulle!")
        self.states[self.length] = state
        self.actions[self.length] = action
        self.rewards[self.length] = reward
        self.done[self.length] = done
        # estimated q values 由模型生成，每个state有多个action，因此 q_values 也有多个
        self.q_values[self.length] = estimated_q
        self.length = self.length + 1

    def update_all(self):
        for i in range(self.capacity-2, -1, -1):
            action = self.actions[i]
            if self.done[i]:
                actual_action_value = 0
            else:
                actual_action_value = self.rewards[i] + self.discount_factor * np.max(self.q_values[i+1])
            estimated_q_value = self.q_values[i, action]
            self.estimate_error[i] = abs(actual_action_value -estimated_q_value)
            self.q_values[i, action] = actual_action_value

    @property
    def is_full(self):
        return self.length >= self.capacity

    def prepare_prob(self, batch_size=128):
        idx = self.estimate_error < self.err_threshold
        self.idx_err_lo = np.squeeze(np.where(idx))
        self.idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))
        prob_err_hi = len(self.idx_err_hi) / self.length
        prob_err_hi = max(prob_err_hi, 0.5)
        self.num_samples_err_hi = int(prob_err_hi * batch_size)
        self.num_samples_err_lo = batch_size - self.num_samples_err_hi

    def random_batch(self):
        idx_lo = np.random.choice(self.idx_err_lo,
                                  size=self.num_samples_err_lo,
                                  replace=False)
        idx_hi = np.random.choice(self.idx_err_hi,
                                  size=self.num_samples_err_hi,
                                  replace=False)
        idx = np.concatenate((idx_lo, idx_hi))
        states_batch = self.states[idx]
        q_values_batch = self.q_values[idx]
        return states_batch, q_values_batch

    def reset(self):
        self.length = 0


if __name__ == "__main__":
    env = gym.make("CartPole")
    state = env.reset()
    state_after, revard, done, info = env.step(1)
    env.render()
    env.close()
