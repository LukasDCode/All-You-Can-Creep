from itertools import product

class Gridsearch():

    def __init__(self, executor, agent_class, **kwargs):
        self.executor = executor
        self.agent_class = agent_class

    def run(self):
        hyperparams = {
                "alpha":[0.001],
                "gamma":[0.99, 0.999, 0.995, 0.9999, 0.99995, 0.99999],
                "entropy_beta":[0.003, 0.001, 0.0005,0.00025, 0.000125],
                "entropy_fall":[1.0001, 1, 0.9999],
                "advantage": ["a2c", "td", "reinforce"],
                "batch_size": [10]
        }
        keys = [*hyperparams.keys()]
        recombinations = product(*[hyperparams[key] for key in keys])

        for param_comb in recombinations:
            params = {key: param_comb[i] for i, key in enumerate(keys)}
            self.executor.submit_task(
                agent_class=self.agent_class,
                **params,
            )



