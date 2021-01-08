class Gridsearch():

    def __init__(self, executor, domain):
        self.executor = executor
        self.domain = domain

    def run(self):

        hyperparams = {
                "alpha":[0.001],
                "gamma":[0.999, 0.9999, 0.99999,1.],
                "entropy":[1,0.5,0.25,0.125,0.075],
                "entropy_fall":[1] #[0.9999,0.999,0.99,0.95,0.9]
            }

        for i in hyperparams["alpha"]:
            for j in hyperparams["gamma"]:
                for k in hyperparams["entropy"]:
                    for l in hyperparams["entropy_fall"]:
                        individual = [i,j,k,l]
                        self.executor.submit_task(params=self.unwrap_params(individual))


    def unwrap_params(self, individual):
        params = {}
        for index, value in enumerate(individual):
            params[self.domain.param_dict()[index][0]] = value
        return params