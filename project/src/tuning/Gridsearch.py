from itertools import product

class Gridsearch():

    def __init__(self, executor, agent_class, **kwargs):
        self.executor = executor
        self.agent_class = agent_class

    def run(self):
        hyperparams = {
                "alpha":[0.001, 0],
                "gamma":[0.99, 0.995, 0.999],
                "entropy_beta":[1e-4, 1e-5, 1e-6],
                "entropy_fall":[1.0001, 1, 0.9999],
                "advantage": ["a2c", "td", "reinforce"],
                "batch_size": [10] 
        }
        keys = [*hyperparams.keys()]
        recombinations = product(*[hyperparams[key] for key in keys])

        for param_comb in recombinations:
            for _ in range(5): 
                params = {key: param_comb[i] for i, key in enumerate(keys)}
                self.executor.submit_task(
                    agent_class=self.agent_class,
                    **params,
                )



