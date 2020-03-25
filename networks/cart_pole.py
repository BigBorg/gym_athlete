import gym
import tensorflow as tf
from networks.base import Athelete

class CartPoleAthelete(Athelete):
    def build_network(self) -> tf.keras.Model:
        input_x = tf.keras.layers.Input(self.state_shape)
        net = tf.keras.layers.Dense(10)(input_x)
        net = tf.keras.layers.Dense(self.action_space)(net)
        model = tf.keras.Model(input_x, net)
        return model

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state = env.reset()
    athelete = CartPoleAthelete("CartPole-v1")
    athelete.simulate(0.3, True)
    print("Done")
