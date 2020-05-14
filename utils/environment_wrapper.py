import gym

class EnvironmentWrapper(object):
    def __init__(self, env_name):
        self.environment = gym.make(env_name)

    def __getattr__(self, item):
        return self.environment.__getattribute__(item)

    def step(self, *args, **kwargs):
        state_after, reward, done, info = self.environment.step(*args, **kwargs)
        height = state_after[0] * (-state_after[2]-1) + state_after[1] * state_after[3]
        return state_after, height, done, info
