class Gridsearch():

    def __init__(self, executor, domain, result_tsv):
        self.executor = executor
        self.domain = domain
        self.result_tsv = result_tsv

    def run(self):

        hyperparams = {
                "alpha":[0.001,0.002,0.004,0.008,0.0016,0.0032,0.0064, 0.0128,0.0256,0.0512, 0.1024,0.2048],
                "gamma":[0.999,0.9995, 0.9999,0.9995, 0.99999,1.],
                "entropy":[1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
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