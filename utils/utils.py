from utils.environment_wrapper import EnvironmentWrapper
import sys, tty, termios

def get_ch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def interact_gym_env(environment_name:str):
    environment = EnvironmentWrapper(environment_name)
    action_space = environment.action_space.n
    state = environment.reset()
    environment.render()
    reward = 0
    print("press {}-{} to interact with env:".format(1, action_space))
    while True:
        action = int(get_ch()) - 1
        _, rev, done, _ = environment.step(action)
        reward += rev
        environment.render()
        print(rev)
        if done:
            break
            print("Done")

    print("Total score: {}".format(reward))
