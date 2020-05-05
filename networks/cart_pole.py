import os
import tensorflow as tf
from networks.base import Athlete

class CartPoleAthelete(Athlete):
    def build_network(self) -> tf.keras.Model:
        input_x = tf.keras.layers.Input(self.state_shape)
        net = tf.keras.layers.Dense(64)(input_x)
        net = tf.keras.layers.LeakyReLU()(net)
        net = tf.keras.layers.Dense(128)(net)
        net = tf.keras.layers.LeakyReLU()(net)
        net = tf.keras.layers.Dense(self.action_space)(net)
        model = tf.keras.Model(input_x, net)
        return model

if __name__ == "__main__":
    # 内存泄露问题
    tf.compat.v1.disable_eager_execution()
    athelete = CartPoleAthelete("CartPole-v1")
    athelete.train(300, os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models/CartPole"))

    # athelete.estimate_model(model_path="/home/borg/githubRepos/gym_athlete/saved_models/CartPole.epoch_55.score_500.h5", render=True)

    # directory = "/home/borg/githubRepos/gym_athlete/saved_models/CartPole/失败"
    # models= sorted([os.path.join(directory, ele) for ele in os.listdir(directory)])
    # for path in models:
        # score = athelete.score_model(model_path=path, num_iteration=20)
        # athelete.estimate_model(model_path=path, render=True)
        # print("Model: ", path)
        # print("Avg score: ", score)athelete.estimate_model(model_path="/home/borg/githubRepos/gym_athlete/saved_models/CartPole.epoch_55.score_500.h5", render=True)
        # input("Continue: ")

    print("Done")
