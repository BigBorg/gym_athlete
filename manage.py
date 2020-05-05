import tensorflow as tf
import argparse

from networks.cart_pole import CartPoleAthelete
from networks.acrobot import AcrobotAthlete
from utils.utils import interact_gym_env

environment_to_class = {
    "CartPole-v1": CartPoleAthelete,
    "Acrobot-v1": AcrobotAthlete
}

argument_parser = argparse.ArgumentParser("Gym Athlete")
argument_parser.add_argument("command", type=str, choices=["train", "eval", "interact"])
argument_parser.add_argument("environment", type=str, choices=["CartPole-v1", "Acrobot-v1"])
argument_parser.add_argument("--model_path", type=str, required=False)
parsed_arguments = argument_parser.parse_args()

tf.compat.v1.disable_eager_execution()
Athlete = environment_to_class[parsed_arguments.environment]
athlete = Athlete(parsed_arguments.environment)

if parsed_arguments.command == "train":
    athlete.train(100, "saved_models/" + parsed_arguments.environment)
elif parsed_arguments.command == "eval":
    if not hasattr(parsed_arguments, "model_path"):
        print("请指定模型路径")
    else:
        athlete.estimate_model(model_path=parsed_arguments.model_path, render=True)
elif parsed_arguments.command == "interact":
    interact_gym_env(parsed_arguments.environment)
