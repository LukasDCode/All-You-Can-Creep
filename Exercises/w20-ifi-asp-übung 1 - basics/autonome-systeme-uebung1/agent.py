import random
import numpy
import copy

"""
 Base class of an autonomously acting and learning agent.
"""
class Agent:

    def __init__(self, params):
        self.nr_actions = params["nr_actions"]

    """
     Behavioral strategy of the agent. Maps states to actions.
    """
    def policy(self, state):
        pass

    """
     Learning method of the agent. Integrates experience into
     the agent's current knowledge.
    """
    def update(self, state, action, reward, next_state, done):
        pass
        

"""
 Randomly acting agent.
"""
class RandomAgent(Agent):

    def __init__(self, params):
        super(RandomAgent, self).__init__(params)
        
    def policy(self, state):
        return random.choice(range(self.nr_actions))

"""
 Autonomous agent using Monte Carlo Rollout Planning.
"""
class MonteCarloRolloutPlanner(Agent):

    def __init__(self, params):
        super(MonteCarloRolloutPlanner, self).__init__(params)
        self.env = params["env"]
        self.gamma = params["gamma"]
        self.horizon = params["horizon"]
        self.simulations = params["simulations"]
        
    def policy(self, state):
        # Tracks Q-values of each action (in the first step)
        Q_values = numpy.zeros(self.nr_actions)
        # Tracks number of each action selections (in the first step).
        action_counts = numpy.zeros(self.nr_actions)
        for _ in range(self.simulations):
            # Copying the current environment provides a simulator for planning.
            generative_model = copy.deepcopy(self.env)
            # TODO Implement planning logic
        return numpy.argmax(Q_values)
