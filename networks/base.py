import time
import random
import gym
import numpy as np
import tensorflow as tf
from utils.replay_memory import ReplayMemory
from utils.motion_tracer import MotionTracer
from utils.environment_wrapper import EnvironmentWrapper


class Athlete(object):
    def __init__(self, environment_name="CartPole-v1", replay_memory_size=10000, action_threshold=0.7, batch_size=64, gamma=0.9):
        self.environment = gym.make(environment_name)
        state = self.environment.reset()
        self.state_shape = state.shape
        self.action_space = self.environment.action_space.n
        self.replay_memory = ReplayMemory(self.state_shape, capacity=replay_memory_size)
        self.model = self.build_network()
        self.target_model = self.build_network()
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
            self.target_model.set_weights(self.model.get_weights())
            self.replay_memory.reset()
            self.simulate(self.action_threshold)
            self.replay_memory.compute_estimated_q(self.target_model, self.gamma)
            num_batches = self.replay_memory.length / self.batch_size
            for j in range(int(num_batches)):
                states, actions, rewards, dones, next_states, estimated_q = self.replay_memory.random_batch(self.batch_size)
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
        reward_count = 0
        while True:
            action = model.predict(state.reshape([1,] + list(self.state_shape)))
            print(state)
            print(action)
            action = np.argmax(action, 1)[0]
            print(action)
            if render:
                time.sleep(0.05)
                self.environment.render()
            state_after, revard, done, _ = self.environment.step(action)
            reward_count += revard
            if done:
                break
            state = state_after

        print("Steps taken: ", reward_count)
        return reward_count

    def score_model(self, model=None, model_path="", num_iteration=10):
        if not model:
            model: tf.keras.Model = tf.keras.models.load_model(model_path)

        scores = []
        for i in range(num_iteration):
            score = self.estimate_model(model)
            scores.append(score)
        avg_score = sum(scores) / num_iteration
        return avg_score

class MotionAthlete(Athlete):
    def __init__(self, environment_name="Acrobot-v1", replay_memory_size=10000, action_threshold=0.7, batch_size=64, gamma=0.9):
        super(MotionAthlete, self).__init__(environment_name, replay_memory_size, action_threshold, batch_size, gamma)
        self.environment.close()
        del self.environment
        self.environment = EnvironmentWrapper(environment_name)
        frame = self.environment.reset()
        frmae_shape = frame.shape
        self.motion_tracer = MotionTracer(frame_shape=frmae_shape)
        self.state_shape =  self.motion_tracer.state_shape
        self.replay_memory = ReplayMemory(self.state_shape, capacity=replay_memory_size)
        del self.model
        del self.target_model
        self.model = self.build_network()
        self.target_model = self.build_network()

    def simulate(self, action_threshold: float):
        print("Simulating...")
        frame = self.environment.reset()
        self.motion_tracer.reset()
        self.motion_tracer.add_frame(frame)
        while not self.replay_memory.is_full:
            state = self.motion_tracer.get_state()
            action = self.choose_action(state, action_threshold)
            frame_after, reward, done, _ = self.environment.step(action)
            self.motion_tracer.add_frame(frame_after)
            state_next = self.motion_tracer.get_state()
            self.replay_memory.add(state, action, reward, done, state_next)
            if done:
                frame = self.environment.reset()
                self.motion_tracer.reset()
                self.motion_tracer.add_frame(frame)
        print("Simulation finished")

        return True

    def estimate_model(self, model=None, model_path="", render=True):
        if not model:
            model: tf.keras.Model = tf.keras.models.load_model(model_path)
        frame = self.environment.reset()
        self.motion_tracer.reset()
        self.motion_tracer.add_frame(frame)
        state = self.motion_tracer.get_state()
        reward_count = 0
        step_count = 0
        while True:
            step_count += 1
            action = model.predict(state.reshape([1,] + list(self.state_shape)))
            print(frame)
            print(action)
            action = np.argmax(action, 1)[0]
            print(action)
            if render:
                time.sleep(0.05)
                self.environment.render()
            frame_after, revard, done, _ = self.environment.step(action)
            reward_count += revard
            if done:
                break
            self.motion_tracer.add_frame(frame_after)
            state = self.motion_tracer.get_state()

        print("Total reward: ", reward_count)
        print("Total step: ", step_count)
        return reward_count