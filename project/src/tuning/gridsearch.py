from .EAsimple import EAsimple
import pandas as pd

params_eaSimple = {
    "current_population": [],
    "generation_max": 200,
    "mutation_rate": 0.003,
    "crossover_rate": 0.8,
    "population_max": 10,
    "name": "EAsimple"
}

eaSimple = EAsimple(params_eaSimple)

eaSimple.run()
iList =[]
jList =[]
kList =[]
result = []
params = params_eaSimple
ea = eaSimple
number_of_execution = 10

for i in [100]:
    params["population_max"] = i
    for j in range(number_of_execution):
        params["current_population"] = []
        ea = EAsimple(params)
        ea.run()
        for k in range(len(ea.winner)):
            average[k] = ea.global_maximum[k] - ea.winner[k] + average[k]
    result.append(sum([o/number_of_execution for o in average])/len(average))
    average = [0 for i in range(params["generation_max"])]
    iList.append(i)
    jList.append(j)
    kList.append(k)


parameters = {"CrossoverRate":iList, "MutationRate":jList, "AverageDistance": kList}
table = pd.DataFrame(parameters)
table.to_excel("GridSearch_eaSimple.xlsx")
print(table)