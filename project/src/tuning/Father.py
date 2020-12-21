import abc
import matplotlib.pyplot as plt
import numpy as np

class Father(abc.ABC):

    def __init__(self, param_dictionary):
        """
        Parameter hier übergeben.
        :param param_dictionary: Dictionary mit einzelnen Parametern für den Algorithmus.
        """
        self.__dict__.update(param_dictionary)
        self.winner = []
        self.global_maximum = []
        self.fitness_list = []
        self.fitness_list_past = []
        self.current_population = []
        self.current_generation = 1
        self.rewards = np.zeros(self.param_dictionary["generation_max"])

    def eval(self, individual):
            rewards_array = [] # array muss noch eingefügt werden über Individual n, slurm log
            winner = max(rewards_array) #alternativ ginge noch der durchschnittliche reward
            stability = self.stability(rewards_array)
            speed = self.convergence_speed(rewards_array)
            return(speed + (stability*2) + winner /4) #prüfen auf gewichtung

    @abc.abstractmethod
    def _new_population(self):
        pass

    @abc.abstractmethod
    def _evaluate(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def _mate(self):
        pass

    @abc.abstractmethod
    def _mutate(self):
        pass

    @abc.abstractmethod
    def _select(self):
        pass

    def output(self):
        for n,m in self.current_population, self.fitness_list:
            rewards_array = []
            gamma = n[0]
            alpha = n[1]
            reward = m #angegebene daten printen lassen

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


