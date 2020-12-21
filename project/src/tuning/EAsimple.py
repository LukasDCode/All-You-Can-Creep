import numpy as np
import random

class EAsimple():

    def __init__(self, executor, domain,  param_dictionary, result_tsv):
        """
        Parameter hier übergeben.
        :param param_dictionary: Dictionary mit einzelnen Parametern für den Algorithmus.
        """
        self.__dict__.update(param_dictionary)
        self.executor = executor
        self.domain = domain
        self.winner = []
        self.global_maximum = []
        self.fitness_list = []
        self.fitness_list_past = []
        self.current_population = []
        self.current_generation = 1
        self.rewards = np.zeros(self.generation_max)
        self.result_tsv = result_tsv


    def run(self):
        self.new_population()
        self.evaluate()
        with open(self.result_tsv, 'w') as tsv:
          for i in range(self.generation_max):
            self.mate()
            self.mutate()
            self.evaluate()
            self.select()
            tsv.write("{}\t{}\n".format(
                self.current_generation, 
                str(zip(self.fitness_list, self.current_population)),
            ))



    def new_population(self):
        self.current_population = []
        for i in range(self.population_max):
            individual = [random.uniform(min,max) for (_,min,max) in self.domain.param_dict()]
            self.current_population.append(individual)
    
    def unwrap_params(self, individual):
        params = {}
        for index, value in enumerate(individual):
            params[self.domain.param_dict()[index][0]] = value
        return params


    def evaluate(self):
        fitness_list = []
        futures = [self.eval(i) for i in self.current_population]
        for future in futures:
            rewards_array = future.get()
            winner = max(rewards_array) #alternativ ginge noch der durchschnittliche reward 
            stability = self.stability(rewards_array)
            speed = self.convergence_speed(rewards_array)
            fitness = (speed + (stability*2) + winner /4) #prüfen auf gewichtung
            self.fitness_list.append(fitness)
        return fitness_list
    
    def eval(self, individual):
        return self.executor.submit_task(self.unwrap_params(individual))

    def mate(self):
        index_current_population_mate = []
        a = 0
        for individual_index in self.current_population:
            index_current_population_mate.append(a)
            a = a+1

        paarungswahrscheinlichkeit = [((1 / np.sqrt(np.pi*0.5)) * np.exp(-(np.square((i /
                            max(self.fitness_list)-1) / 0.5))))for i in self.fitness_list]

        mothers = np.random.choice(a=index_current_population_mate, size=random.randint(0, self.population_max), replace=False,
                                   p=[i/sum(paarungswahrscheinlichkeit) for i in paarungswahrscheinlichkeit])

        for mother in sorted(mothers, reverse=True):
            if np.random.rand() < self.crossover_rate:
                father = self.current_population[random.randint(0, len(self.current_population)-1)]
                child = self.current_population[mother][:1] + father[1:]
                if child not in self.current_population:
                    self.current_population.append(child)

    def mutate(self):
        new_current_population = []
        for individuum in self.current_population:
            new_individuum = []
            for index, param in enumerate(individuum):
                if np.random.rand() < self.mutation_rate:
                    (_,min,max) = self.domain.param_dict[index]
                    new_param = random.uniform(min,max) 
                    new_individuum.append(new_param)
                else:
                    new_individuum.append(param)
            new_current_population.append(new_individuum)
        self.current_population = new_current_population

    def select(self):
        index_current_population_select = []
        a = 0
        for individual_index in self.current_population:
            index_current_population_select.append(a)
            a = a+1

        sterbewahrscheinlichkeit = [((1 / np.sqrt(np.pi * 0.5)) * np.exp(-(np.square((i / max(self.fitness_list))/0.5)))) for i in
                                    self.fitness_list]

        index_current_population_select = np.random.choice(a=index_current_population_select,
                                                           size=len(self.current_population) - self.population_max,
                                                           replace=False, p=[i/sum(sterbewahrscheinlichkeit) for i in sterbewahrscheinlichkeit])

        for index in sorted(index_current_population_select, reverse=True):
            del self.current_population[index]

        self.current_generation = self.current_generation + 1

    def convergence_speed(self, rewards):
        conSpeed = sum(self.rewards/len(rewards)) * max(rewards).index() #durchschnittliche Steigung
        return conSpeed

    def stability(self, rewards):
                for k in range(len(rewards)):
                    try:
                        self.rewards[k] = rewards[k] - rewards[k - 1]
                    except IndexError:
                        self.rewards[k] #negativ bei abfall, positiv bei steigung
                AvgStability = sum([i for i in self.rewards]) / len(rewards)
                return AvgStability
