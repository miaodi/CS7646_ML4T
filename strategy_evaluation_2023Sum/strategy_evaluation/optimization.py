import ManualStrategy
import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt
import marketsimcode
import indicators
import pygad
from enum import Enum

num_generations = 50  # Number of generations.
# Number of solutions to be selected as parents in the mating pool.
num_parents_mating = 5
# To prepare the initial population, there are 2 ways:
# 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
# 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
sol_per_pop = 50  # Number of solutions in the population.

last_fitness = 100000


def callback_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(
        generation=ga_instance.generations_completed))
    print("Solution   = {solution}".format(
        solution=ga_instance.best_solution()[0]))
    print("Fitness    = {fitness}".format(
        fitness=ga_instance.best_solution()[1]))
    print("Change     = {change}".format(
        change=ga_instance.best_solution()[1] - last_fitness))
    last_fitness = ga_instance.best_solution()[1]
    print(ga_instance.population)


class Types(Enum):
    BB = 1
    SMA = 2
    RSI = 3
    SO = 4
    Weight = 5


Type = Types.Weight
if Type == Types.BB:
    signal_weight = [0, 1, 0, 0]

    def fitness_func(ga_instance, solution, soolution_idx):
        ms = ManualStrategy.ManualStrategy(
            'JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
        ms.setBB(solution[0], solution[1])
        ms.initialize()
        signal = ms.Signal(signal_weight)
        res = ms.portValue()
        return res.tail(1).values[0][0]

    gene_space = [
        range(2, 40),
        {'low': .2, 'high': 3},
    ]
    num_genes = 2
elif Type == Types.SMA:
    signal_weight = [1, 0, 0, 0]

    def fitness_func(ga_instance, solution, soolution_idx):
        ms = ManualStrategy.ManualStrategy(
            'JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
        ms.setSMA(solution[0], solution[0]+solution[1])
        ms.initialize()
        signal = ms.Signal(signal_weight)
        res = ms.portValue()
        return res.tail(1).values[0][0]

    gene_space = [
        range(5, 30),
        range(1, 50)
    ]
    num_genes = 2
elif Type == Types.RSI:
    signal_weight = [0, 0, 1, 0]

    def fitness_func(ga_instance, solution, soolution_idx):
        ms = ManualStrategy.ManualStrategy(
            'JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
        ms.setRSI(solution[0], solution[1])
        ms.initialize()
        signal = ms.Signal(signal_weight)
        res = ms.portValue()
        return res.tail(1).values[0][0]

    gene_space = [
        range(5, 30),
        range(10, 40)
    ]
    num_genes = 2
elif Type == Types.SO:
    signal_weight = [0, 0, 0, 1]

    def fitness_func(ga_instance, solution, soolution_idx):
        ms = ManualStrategy.ManualStrategy(
            'JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
        ms.setSO(solution[0], solution[1], solution[2])
        ms.initialize()
        signal = ms.Signal(signal_weight)
        res = ms.portValue()
        return res.tail(1).values[0][0]

    gene_space = [
        range(7, 28),
        range(1, 21),
        range(1, 21)
    ]
    num_genes = 3
elif Type == Types.Weight:
    def fitness_func(ga_instance, solution, soolution_idx):
        ms = ManualStrategy.ManualStrategy(
            'JPM', dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
        ms.setSMA(10, 34)
        ms.setRSI(13, 19)
        ms.setSO(20, 6, 1)
        ms.setBB(16, 1.47924663)
        ms.initialize()
        signal = ms.Signal(solution)
        res = ms.portValue()
        return res.tail(1).values[0][0]

    gene_space = [
        {'low': .0, 'high': 1},
        {'low': .0, 'high': 1},
        {'low': .0, 'high': 1},
        {'low': .0, 'high': 1},
    ]
    num_genes = 4


# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       crossover_type="scattered",
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       mutation_type="random",
                       parent_selection_type='tournament',
                       on_generation=callback_generation,
                       mutation_num_genes=num_genes,
                       gene_space=gene_space)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_fitness()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(
    solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(
    solution_idx=solution_idx))

prediction = fitness_func(solution)
print("Predicted output based on the best solution : {prediction}".format(
    prediction=prediction))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(
        best_solution_generation=ga_instance.best_solution_generation))

# Saving the GA instance.
# The filename to which the instance is saved. The name is without extension.
filename = 'genetic'
ga_instance.save(filename=filename)
