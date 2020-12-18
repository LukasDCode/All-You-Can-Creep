from .EAsimple import EAsimple

params_eaSimple = {
    "generation_max": 50,
    "mutation_rate": 0.05,
    "crossover_rate": 0.8,
    "population_max": 10,
    "name": "EAsimple"
}


evolAlg = EAsimple(params_eaSimple)
evolAlg.run()





