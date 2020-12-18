from .Father import Father
import numpy as np
import random

class EAsimple(Father):

    def run(self):
        self._new_population()
        self._evaluate()
        for i in range(self.generation_max):
            self._mate()
            self._mutate()
            self._evaluate()
            self._select()
        self.convergence_speed()

    def _new_population(self):
        self.current_population = []
        for i in range(self.population_max):
            gamma = random.uniform(0.9, 1)
            alpha = random.uniform(0.00001, 0.2)
            self.current_population.append([gamma, alpha])

    def _evaluate(self):
        fitness_list = []
        for n in self.current_population:
            fitness_list.append(self.eval(n))
        self.fitness_list = fitness_list

    def _mate(self):
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

    def _mutate(self):
        new_current_population = []
        for position in self.current_population:
            new_position = []
            for coordinate in position:
                if np.random.rand() < self.mutation_rate:
                    if 0 == position.index(coordinate):
                        new_coordinate = random.uniform(0.9, 1)
                    else:
                        new_coordinate = random.uniform(0.00001, 0.2)
                    new_position.append(new_coordinate)
                else:
                    new_position.append(coordinate)
            new_current_population.append(new_position)
        self.current_population = new_current_population

    def _select(self):
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
        #nach jeder generation soll die liste der rewards + settingsbeschreibung + bewertung gespeichert werden


