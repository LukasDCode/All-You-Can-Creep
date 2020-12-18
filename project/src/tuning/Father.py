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

        self.generation_change = 0
        self.current_generation = 1

    def eval(self, individual):
            rewards_array = [] # array muss noch eingefügt werden
            winner = max(rewards_array)
            speed = self.convergence_speed(rewards_array)
            stability = self.stability(rewards_array)
            return(speed* (stability*2) *winner /4)

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

    def convergence_speed(self, rewards):
        conSpeed = 0 #ableitung quadrieren, negativ VZ und positiv VZ Fallunterscheidung
        for n in range(201):
            if n % 20 != 0 and m != n:
                zaehler = (max(self.winner[m:n]) - self.winner[m]) + zaehler
            else:
                conSpeed_m = zaehler / (20 * (self.global_maximum[m] - self.winner[m]))
                conSpeed = conSpeed + conSpeed_m
                conSpeed_m = 0
                zaehler = 0
                if n > 0 and m < 180:
                    m = m + 20
        self.avg_con_speed = 1 / 9 * conSpeed
        return self.avg_con_speed

    def stability(self, rewards):
                Stability = [0 for i in range(self.param_dictionary["generation_max"])]
                for k in range(len(rewards)):
                    Stability[k] = rewards[k] / rewards[k - 1]
                AvgStability = sum([i for i in Stability]) / len(rewards)
                return AvgStability


