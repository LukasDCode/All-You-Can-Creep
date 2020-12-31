class Gridsearch():

    def __init__(self, executor, domain, result_tsv):
        self.executor = executor
        self.domain = domain
        self.result_tsv = result_tsv

    def run(self):

        hyperparams = {
                "alpha":[0.001],
                "gamma":[0.9,1],
                "entropy":[1e-323, 1e-4]
            }

        for i in hyperparams["alpha"]:
            for j in hyperparams["gamma"]:
                for k in hyperparams["entropy"]:
                    individual = [i,j,k]
                    self.executor.submit_task(self.unwrap_params(individual))


    def unwrap_params(self, individual):
        params = {}
        for index, value in enumerate(individual):
            params[self.domain.param_dict()[index][0]] = value
        return params