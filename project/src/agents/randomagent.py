import numpy

from .agent import Agent

class RandomAgent(Agent):

    def __init__(self, env, **kwargs):
        super().__init__(env)

    def policy(self, state):
        random_m = numpy.random.uniform(-1, 1, self.nr_actions)
        return random_m

    @staticmethod
    def agent_name():
        raise NotImplemented()