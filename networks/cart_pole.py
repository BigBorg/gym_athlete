import os
import tensorflow as tf
from networks.base import Athlete

class CartPoleAthelete(Athlete):
    def build_network(self) -> tf.keras.Model:
        input_x = tf.keras.layers.Input(self.state_shape)
        net = tf.keras.layers.Dense(64, activation="softmax")(input_x)
        net = tf.keras.layers.Dense(64, activation="softmax")(net)
        net = tf.keras.layers.Dense(self.action_space)(net)
        model = tf.keras.Model(input_x, net)
        return model

if __name__ == "__main__":
    # 内存泄露问题
    tf.compat.v1.disable_eager_execution()
    athelete = CartPoleAthelete("CartPole-v1")
    # athelete.train(200, os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models/CartPole"))
    athelete.estimate_model(model_path="/home/borg/githubRepos/gym_athlete/saved_models/CartPole.epoch_95.score_500.h5", render=True)
    print("Done")
