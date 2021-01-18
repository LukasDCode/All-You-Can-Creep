class Agent:
    """
    Base class of an autonomously acting and learning agent.
    """

    @staticmethod
    def hyper_params():
        """ Returns list of tuples (name, min, max)"""
        return {
            "name": {"min": 0, "max": 1},
        }

    @staticmethod
    def agent_name():
        raise NotImplemented()

    def __init__(self, env):
        self.__dict__.update({
            "env": env,
            "nr_input_features": env.observation_space.shape[-1],
            "nr_actions": env.action_space.shape[0],
            "lower_bound": env.action_space.low[0],
            "upper_bound": env.action_space.high[0],
        })

    """
     Behavioral strategy of the agent. Maps states to actions.
    """
    def policy(self, state):
        pass

    """
     Learning method of the agent. Integrates experience into
     the agent's current knowledge.
    """
    def update(self,nr_episode, state, action, reward, next_state, done):
        return {}

    def state_dict(self):
        return {}


    def load_state_dict(self, state_dict, only_model, strict=False):
        pass
