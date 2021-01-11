"""
 Base class of an autonomously acting and learning agent.
"""
class Agent:

    """ Returns list of tuples (name, min, max)"""
    @staticmethod
    def hyper_params():
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
        })
        pass

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
