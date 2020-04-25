import gym
import numpy as np

class ReplayMemory(object):
    def __init__(self, state_shape:tuple, capacity=10000):
        self.states = np.zeros([capacity] + list(state_shape), dtype=float)
        self.actions = np.zeros((capacity,), dtype=int)
        self.rewards = np.zeros((capacity,), dtype=float)
        self.done = np.zeros((capacity,), dtype=bool)
        self.next_states = np.zeros([capacity] + list(state_shape), dtype=float)
        self.length = 0
        self.capacity = capacity

    def add(self, state, action, reward, done, next_state):
        if self.is_full:
            raise Exception("replay memory is full!")
        self.states[self.length] = state
        self.actions[self.length] = action
        self.rewards[self.length] = reward
        self.done[self.length] = done
        self.next_states[self.length] = next_state
        self.length = self.length + 1

    @property
    def is_full(self):
        return self.length >= self.capacity

    def random_batch(self, batch_size):
        idx = np.random.choice(range(self.length),
                                  size=batch_size,
                                  replace=False)
        states = self.states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.done[idx]
        next_states = self.next_states[idx]
        return states, actions, rewards, dones, next_states

    def reset(self):
        self.length = 0


if __name__ == "__main__":
    import time
    env = gym.make("CartPole-v1")
    state = env.reset()
    while True:
        state_after, revard, done, info = env.step(1)
        if done:
            print("DEBUG")
            env.reset()
        env.render()
        time.sleep(0.1)
    env.close()
