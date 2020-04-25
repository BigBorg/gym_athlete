import os
import tensorflow as tf
from networks.base import Athlete

class CartPoleAthelete(Athlete):
    def build_network(self) -> tf.keras.Model:
        input_x = tf.keras.layers.Input(self.state_shape)
        net = tf.keras.layers.Dense(64, activation='relu')(input_x)
        net = tf.keras.layers.Dense(32, activation='relu')(net)
        net = tf.keras.layers.Dense(self.action_space)(net)
        model = tf.keras.Model(input_x, net)
        return model

if __name__ == "__main__":
    # 内存泄露问题
    tf.compat.v1.disable_eager_execution()
    athelete = CartPoleAthelete("CartPole-v1")
    athelete.train(100, os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models/CartPole"))

    # athelete.estimate_model(model_path="/home/borg/githubRepos/gym_athlete/saved_models/CartPole.epoch_55.score_500.h5", render=True)

    # models= [                                                              /home/borg/githubRepos/gym_athlete/saved_models/CartPole.epoch_30.score_500.h5
    #     # r"/home/borg/githubRepos/gym_athlete/saved_models/CartPole.epoch_20.score_500.h5",
    #     r"/home/borg/githubRepos/gym_athlete/saved_models/CartPole.epoch_25.score_500.h5",
    #     # r"/home/borg/githubRepos/gym_athlete/saved_models/CartPole.epoch_35.score_500.h5"
    # ]
    # for path in models:
    #     score = athelete.score_model(model_path=path, num_iteration=20)
    #     print("Model: ", path)
    #     print("Avg score: ", score)
    #     input("Continue: ")

    print("Done")
