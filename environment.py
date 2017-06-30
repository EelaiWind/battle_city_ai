import gym

class Environment():
    def __init__(self, env_name):
        self._env_name = env_name
        self._env = gym.make(env_name)

    def copy(self):
        return Environment(self._env_name)

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)
