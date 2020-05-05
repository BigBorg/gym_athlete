import os
import tensorflow as tf
from networks.base import MotionAthlete

class AcrobotAthlete(MotionAthlete):
    def build_network(self) -> tf.keras.Model:
        # 网络模型摘自： https://github.com/FelixNica/OpenAI_Acrobot_D3QN/blob/8074bc276cb114a46684bba12c47f8c9d373fbd3/dueling_dqn_keras.py#L9
        inputs = tf.keras.layers.Input(shape=self.state_shape)

        x = tf.keras.layers.Dense(512, activation='relu')(inputs)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)

        V = tf.keras.layers.Dense(1, activation=None)(x)
        A = tf.keras.layers.Dense(self.action_space, activation=None)(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return tf.keras.Model(inputs=inputs, outputs=Q)


if __name__ == "__main__":
    # 内存泄露问题
    tf.compat.v1.disable_eager_execution()
    athelete = AcrobotAthlete("Acrobot-v1", action_threshold=0.9)
    athelete.train(300, os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models/CartPole"))

    print("Done")