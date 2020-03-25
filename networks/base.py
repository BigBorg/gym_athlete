import random
import gym
import numpy as np
import tensorflow as tf
from utils.replay_memory import ReplayMemory


class Athelete(object):
    def __init__(self, environment_name="CartPole-v1"):
        self.environment = gym.make(environment_name)
        state = self.environment.reset()
        self.state_shape = state.shape
        self.action_space = self.environment.action_space.n
        self.replay_memory = ReplayMemory(self.state_shape, self.action_space)
        self.model = self.build_network()

    def build_network(self) -> tf.keras.Model:
        pass

    def choose_action(self, state:np.ndarray, threshold:float):
        if random.random() > threshold:
            # 随机取结果
            action = random.randint(0, self.action_space - 1)
        else:
            # 模型取结果
            results = self.model.predict(state.reshape([1] + list(state.shape)))
            action = np.argmax(results,1)[0]
        return action

    def simulate(self, action_threshold:float, verbose=False):
        state = self.environment.reset()
        while not self.replay_memory.is_full:
            action = self.choose_action(state, action_threshold)
            state_after, revard, done, _ = self.environment.step(action)
            estimated_q = self.model.predict(state.reshape([1] + list(state.shape)))
            self.replay_memory.add(state, action, revard, done, estimated_q[0])
            state = state_after
            if verbose:
                self.environment.render()
            if done:
                state = self.environment.reset()

        return True
