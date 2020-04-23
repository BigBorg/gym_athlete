import time
import random
import gym
import numpy as np
import tensorflow as tf
from utils.replay_memory import ReplayMemory


class Athlete(object):
    def __init__(self, environment_name="CartPole-v1", replay_memory_size=10000, action_threshold=0.7, batch_size=64, gamma=0.9):
        self.environment = gym.make(environment_name)
        state = self.environment.reset()
        self.state_shape = state.shape
        self.action_space = self.environment.action_space.n
        self.replay_memory = ReplayMemory(self.state_shape, capacity=replay_memory_size)
        self.model = self.build_network()
        self.action_threshold= action_threshold
        self.batch_size = batch_size
        self.gamma = gamma

    def build_network(self) -> tf.keras.Model:
        yield NotImplemented()

    def choose_action(self, state:np.ndarray, threshold:float):
        if random.random() > threshold:
            # 随机取结果
            action = random.randint(0, self.action_space - 1)
        else:
            # 模型取结果
            results = self.model.predict(state.reshape([1] + list(state.shape)))
            action = np.argmax(results,1)[0]
        return action

    def simulate(self, action_threshold:float):
        state = self.environment.reset()
        while not self.replay_memory.is_full:
            action = self.choose_action(state, action_threshold)
            state_after, reward, done, _ = self.environment.step(action)
            self.replay_memory.add(state, action, reward, done, state_after)
            state = state_after
            if done:
                state = self.environment.reset()

        return True

    def train(self, epoch=100, model_prefix="saved_models/model"):
        model_prefix = model_prefix + ".epoch_{}.score_{}.h5"
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=tf.losses.mean_squared_error)
        for i in range(epoch):
            print("Epoch {} running:...".format(i))
            self.simulate(self.action_threshold)
            num_batches = self.replay_memory.length / self.batch_size
            for j in range(int(num_batches)):
                states, actions, rewards, dones, next_states = self.replay_memory.random_batch(self.batch_size)
                estimated_q = self.model.predict(states)
                estimated_next_q = self.model.predict(next_states)
                for index in range(self.batch_size):
                    if not dones[index]:
                        estimated_q[index] = rewards[index] + self.gamma * estimated_next_q[index]
                    else:
                        estimated_q[index] = rewards[index]

                self.model.fit(states, estimated_q, epochs=1, verbose=0)

            if i % 5 == 0:
                score = self.estimate_model(self.model, render=False)
                model_path = model_prefix.format(i, score)
                print("Saving model: {} ...".format(model_path))
                self.model.save(model_prefix.format(i, score))

    def estimate_model(self, model=None, model_path="", render=True):
        if not model:
            model: tf.keras.Model = tf.keras.models.load_model(model_path)
        state = self.environment.reset()
        step_count = 0
        while True:
            step_count += 1
            action = model.predict(state.reshape((1,4)))
            print(state)
            print(action)
            action = np.argmax(action, 1)[0]
            print(action)
            if render:
                time.sleep(0.1)
                self.environment.render()
            state_after, revard, done, _ = self.environment.step(action)
            if done:
                break
            state = state_after

        self.environment.close()
        print("Steps taken: ", step_count)
        return step_count
