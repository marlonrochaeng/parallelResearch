import random
import numpy as np

def create_individual(ET, individual=None, heuristic=None):
    individual = individual 
    if individual is None:
        individual = generate_random_individual(ET)
    fitness = get_fitness(ET, individual)
    heuristic = heuristic
    return individual, fitness, heuristic

def generate_random_individual(ET):
    return np.random.randint(0, ET.shape[1], ET.shape[0])

def get_max_in_array(array):
    '''
    Esta função a posição do menor elemento do array passado por parametro
    '''
    return np.where(array == array.max())[0][0]

def get_min_in_array(maquinas):
    '''
    Esta função a posição do menor elemento do array passado por parametro
    '''
    return np.where(maquinas == maquinas.min())[0][0]

def get_fitness(ET, individual):
    maquinas = np.zeros(ET.shape[1])
    
    #print("individual:", individual)

    for i in range(ET.shape[0]):
        
        maquinas[individual[i]] += ET[i][individual[i]]
    
    maquinas = maquinas
    return maquinas[get_max_in_array(maquinas)]