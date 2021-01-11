import numpy

from .agent import Agent

class RandomAgent(Agent):

    def __init__(self, params):
        super().__init__(params)

    def policy(self, state):
        random_m = numpy.random.uniform(-1, 1, 9) 
        return random_m
