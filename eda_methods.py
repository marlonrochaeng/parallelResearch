import numpy.random as npr
import random
import numpy as np
import individual
from utils import Utils
import json
from joblib import Parallel, delayed


def select_with_choice(population):
    max = sum([c for c in population])
    selection_probs = [c/max for c in population]
    return npr.choice(len(population), p=selection_probs)

def tournament_selection(lista):
    k = int(len(lista)/3)

    tournament_list = []
    enum_list = list(enumerate(lista))

    for _ in range(k):
        tournament_list.append(random.choice(enum_list))
    
    tournament_list = sorted(tournament_list, key=lambda x:x[1], reverse=True)
    return tournament_list[0][0]

def create_prob_matrix(jobs, machines):
    return np.zeros((jobs, machines))

def fill_prob_matrix(prob_matrix, individuals):
    for i in individuals:
        for j in range(len(i.individual)):
            prob_matrix[j][i.individual[j]] += 1
    return prob_matrix

def fill_single_prob_matrix(prob_matrix, individual):
    for j in range(len(individual.individual)):
        prob_matrix[j][individual.individual[j]] += 1
    return prob_matrix

def paralel_gen(ET, prob_matrix, selection_method):
    new_gen =individual.create_individual(ET, create_new_individual(prob_matrix, selection_method))
    return new_gen

def create_new_individual(prob_matrix, selection_method):
    new_individual = np.random.randint(0, 0, 0)
    count = 0

    for i in prob_matrix:
        count += 1
        if selection_method == 'roulette wheel':
            choice = select_with_choice(i)
        elif selection_method == 'tournament':
            choice = tournament_selection(i)       

        new_individual = np.append(new_individual, choice)

    return new_individual

def create_first_gen(path, jobs, machines):
    gen = []
    
    u = Utils()
    ET, CT, maquinas = u.initialize(path, jobs, machines)

    res, individuos = u.maxmin2(ET, CT, maquinas)
    gen.append(individual.create_individual(ET.copy(), individuos, 'maxmin'))

    res, individuos = u.minmin(ET, CT, maquinas)
    gen.append(individual.create_individual(ET.copy(), individuos, 'minmin'))

    res, individuos = u.mct2(ET, CT, maquinas)
    gen.append(individual.create_individual(ET.copy(), individuos, 'mct'))

    res, individuos = u.met(ET, CT, maquinas)
    gen.append(individual.create_individual(ET.copy(), individuos, 'met'))

    res, individuos = u.olb(ET, CT, maquinas)
    gen.append(individual.create_individual(ET.copy(), individuos, 'olb'))
    
    #for _ in range(numInd):
    #    gen.append(individual.create_individual(ET.copy()))

    #save_to_json(100)

    pop = json.load(open('populations/population_100.json'))

    #população gerada por força bruta
    
    _path = path.split('/')[1].split('.')[0]
    print(_path)
    u_c_pop = json.load(open(f'populations/population_map-{_path}.txt.json'))
    print(f"--------population_map-{_path}.txt.json---------")

    for i in range(len(u_c_pop)):
        gen.append(individual.create_individual(ET.copy(), u_c_pop[str(i)]))
    
    #população gerada atraves da populacao controle
    for i in range(len(pop)):
        gen.append(individual.create_individual(ET.copy(), pop[str(i)]))
    
    first_gen_len = len(gen)

    return gen

def order_pop(arr):
    return sorted(arr)
    