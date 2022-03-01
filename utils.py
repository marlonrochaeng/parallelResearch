import numpy as np
import json
from numba import jit
import random
import numpy.random as npr
import pandas as pd
import os.path as path
import copy
from joblib import Parallel, delayed


def get_initial_pop():
  pop = json.load(open('populations/population_100.json'))
  pop = [pop[p] for p in pop]
  return np.array(pop)
  
def get_big_pop(path):
  _path = path.split('.')[0].replace('512x16/', '')
  u_c_pop = json.load(open(f'populations/population_map-{_path}.txt.json'))
  u_c_pop = [u_c_pop[p] for p in u_c_pop]
  return np.array(u_c_pop)

@jit(nopython=True)
def mutate(individuals, mut_prob, num_machines):
  for i in individuals:
    p = random.randint(1,100)
    if p <= mut_prob:
      pos = random.randint(0,len(i)-1)
      i[pos] = random.randint(0, num_machines - 1)

def order_pop(ET, individuals):
  makespans = []
  makespans = Parallel(n_jobs=8)(delayed(get_fitness)(ET, i) for i in individuals)
  makespans = np.array(makespans)
  inds = makespans.argsort()
  return individuals[inds]

def create_prob_matrix(jobs, machines):
  return np.zeros((jobs, machines), dtype=int)

@jit(nopython=True)
def fill_prob_matrix(prob_matrix, individuals):
        for i in individuals:
            for j in range(len(i)):
                prob_matrix[j][i[j]] += 1

def get_fitness(ET, individual):
    maquinas = np.zeros(ET.shape[1])
    
    for i in range(ET.shape[0]):
        
        maquinas[individual[i]] += ET[i][individual[i]]
    
    maquinas = maquinas
    return maquinas[get_max_in_array(maquinas)]

def get_min_in_matrix(matrix):
  '''
  Esta função retorna a linha e coluna do menor elemento da matriz passada por parametro
  '''
  return np.unravel_index(matrix.argmin(), matrix.shape)

def get_max_in_matrix(matrix):
  '''
  Esta função retorna a linha e coluna do menor elemento da matriz passada por parametro
  '''
  return np.unravel_index(matrix.argmax(), matrix.shape)

def get_min_in_array(array):
  '''
  Esta função a posição do menor elemento do array passado por parametro
  '''
  return np.where(array == array.min())[0][0]

def get_max_in_array(array):
  '''
  Esta função a posição do menor elemento do array passado por parametro
  '''
  return np.where(array == array.max())[0][0]

def minmin(ET,CT, maquinas):
  individuo = [np.inf for i in range(ET.shape[0])]
  pos = 0
  et_copy = ET.copy()
  while ET.shape[0] != 0:
    #print("et_shape", ET.shape)
    
    min_row, min_col = get_min_in_matrix(CT)
    maquinas[min_col] += ET[min_row][min_col]

    for i in range(ET.shape[0]):
      CT[i][min_col] += ET[min_row][min_col]#maquinas[min_col]
    
    for i in range(len(et_copy)):
      if (ET[min_row] == et_copy[i]).all():
        pos = i
        break

    ET = np.delete(ET,(min_row),0)
    CT = np.delete(CT,(min_row),0)
    individuo[pos] = min_col

  return maquinas, individuo

def maxmin2(ET,CT, maquinas):
  individuo = [np.inf for i in range(ET.shape[0])]
  pos = 0
  et_copy = ET.copy()
  while ET.shape[0] != 0:
    mins = []
    for i in range(ET.shape[0]):
      mins.append((i, get_min_in_array(CT[i]),CT[i][get_min_in_array(CT[i])]))
    
    #print("mins:", mins)
    #print("max of mins", max(mins, key=lambda item:item[1]))
    
    max_of_mins = max(mins, key=lambda item:item[2])
    min_exec = get_min_in_array(CT[max_of_mins[0]])

    maquinas[max_of_mins[1]] += ET[max_of_mins[0]][min_exec]

    for i in range(ET.shape[0]):
      CT[i][max_of_mins[1]] += ET[max_of_mins[0]][min_exec]

    for i in range(len(et_copy)):
      if (ET[max_of_mins[0]] == et_copy[i]).all():
        pos = i
        break 

    individuo[pos] = min_exec

    ET = np.delete(ET,(max_of_mins[0]),0)
    CT = np.delete(CT,(max_of_mins[0]),0)

  return maquinas, individuo


def mct2(ET,CT, maquinas):
  #o correto
  individuo = [np.inf for i in range(ET.shape[0])]
  pos = 0
  
  while ET.shape[0] != 0:  
    temp = maquinas.copy()

    for i in range(len(temp)):
      temp[i] += ET[0][i]

    menor = get_min_in_array(temp)

    maquinas[menor] += ET[0][menor]
    individuo[pos] = menor
    pos += 1

    ET = np.delete(ET,(0),0)
  

  return maquinas, individuo

def met(ET,CT, maquinas):
  individuo = [np.inf for i in range(ET.shape[0])]
  pos = 0

  while ET.shape[0] != 0:  

    menor = get_min_in_array(ET[0])

    maquinas[menor] += ET[0][menor]

    ET = np.delete(ET,(0),0)
    individuo[pos] = menor
    pos += 1

  return maquinas, individuo

def olb(ET,CT, maquinas):
  individuo = [np.inf for i in range(ET.shape[0])]
  pos = 0

  while ET.shape[0] != 0:  
    temp = maquinas.copy()

    #for i in range(len(temp)):
    #  temp[i] += ET[0][i]

    menor = get_min_in_array(temp)

    maquinas[menor] += ET[0][menor]
    individuo[pos] = menor
    pos += 1

    ET = np.delete(ET,(0),0)

  return maquinas, individuo

def sufferage(self, ET,CT, maquinas):
  ets = []
  sufferage = {}
  cts = []
  maquinas_cp = []

  while ET.shape[0] != 0:  
    maquinas_v = [False for i in range(len(maquinas))] 

    maquinas_op = {
      i: [] for i in range(len(maquinas))
    }
    #ultima daquela maquina, nao ultima escalonada
    for i in range(ET.shape[0]):
      
      ets.append(ET.copy())
      cts.append(CT.copy())
      maquinas_cp.append(maquinas.copy())

      temp = CT.copy()

      fst_minimum = self.get_min_in_array(temp[i])
      temp[i][fst_minimum] = np.inf
      scd_minimum = self.get_min_in_array(temp[i])

      #print("Value1: ", CT[i][fst_minimum])
      #print("Value2: ", CT[i][scd_minimum])

      sufferage[i] = CT[i][scd_minimum] - CT[i][fst_minimum]

      #sufferages.append(sufferage)
      print("Sufferage: ", sufferage)

      
      #input()

      if not maquinas_v[fst_minimum]:
        maquinas[fst_minimum] += ET[i][fst_minimum]
        maquinas_v[fst_minimum] = True

        maquinas_op[fst_minimum].append(i)

        for j in range(ET.shape[0]):
          CT[j][fst_minimum] += ET[i][fst_minimum]
        
        ET = np.delete(ET,(i),0)
        CT = np.delete(CT,(i),0)
        
      else:
        if sufferage[maquinas_op[fst_minimum][-1]] < sufferage[i]:
          
          ET = ets[-2].copy()
          CT = cts[-2].copy()

          i = i+1

          #fst_minimum = self.get_min_in_array(CT[i])
          maquinas = maquinas_cp[-1].copy()

          maquinas[fst_minimum] += ET[i][fst_minimum]
          for j in range(ET.shape[0]):
            CT[j][fst_minimum] += ET[i][fst_minimum]
          
          
          
          ET = np.delete(ET,(i),0)
          CT = np.delete(CT,(i),0)

        print("i:", i)
        print("Tamanho: ", ET.shape[0])
        print("Maquinas: ", maquinas)
          

    return maquinas

def select_with_choice(population):
  max = sum([c for c in population])
  selection_probs = [c/max for c in population]
  return npr.choice(len(population), p=selection_probs)

def create_new_individual(prob_matrix):
  new_individual = np.random.randint(0, 0, 0)
  count = 0

  for i in prob_matrix:
    count += 1
    choice = select_with_choice(i)
    new_individual = np.append(new_individual, choice)
  
  return new_individual

def create_new_pop(prob_matrix, qtde):
  individuals = []
  for i in range(qtde):
    individuals.append(create_new_individual(prob_matrix))
  return np.array(individuals)

def create_new_pop_parallel(prob_matrix):
  return np.array(create_new_individual(prob_matrix))

def add_heuristics(ET, CT, n_resources):
  heuristics = []
  heuristics.append(maxmin2(ET, CT, np.zeros(n_resources,dtype=float))[1])
  heuristics.append(minmin(ET, CT, np.zeros(n_resources,dtype=float))[1])
  heuristics.append(mct2(ET, CT, np.zeros(n_resources,dtype=float))[1])
  heuristics.append(met(ET, CT, np.zeros(n_resources,dtype=float))[1])
  heuristics.append(olb(ET, CT, np.zeros(n_resources,dtype=float))[1])
  return np.array(heuristics)

def save_to_csv(jobs, machines, numInd, numGen, best_makespan,
  to_matrix, exec_time, selection_method, elitism, i_path, mutation):
  path_ = 'resultados/eda.csv'
  if path.exists(path_):
    df_results = pd.read_csv(path_, header=0, index_col=0)
  else:
    columns = ['jobs','machines','numInd','numGen','makespan', 'to_matrix_percentage']
    df_results = pd.DataFrame(columns=columns)

  df_results = df_results.append(
      {'jobs': jobs,
        'machines': machines,
        'numInd': numInd,
        'numGen': numGen,
        'makespan': best_makespan,
        'to_matrix_percentage': to_matrix,
        'exec_time': exec_time,
        'selection_method': selection_method,
        'elitismo': elitism,
        'instance': i_path,
        'mutation':mutation}, 
                  ignore_index=True)   
  df_results.to_csv(path_)     
  df_results = df_results.loc[:, ~df_results.columns.str.contains('^Unnamed')]

def stochastic_2_opt(ET, city_tour):
  best_route = copy.deepcopy(city_tour)      
  i, j  = random.sample(range(0, len(city_tour)-1), 2)          
  best_route[i], best_route[j]  = best_route[j], best_route[i]           
  makespan = get_fitness(ET, best_route)                   
  return best_route, makespan
    
# Function: Local Search
def local_search(ET, city_tour, max_attempts = 50, neighbourhood_size = 5):
  count = 0
  solution = copy.deepcopy(city_tour)
  sol_makespan = get_fitness(ET,solution)
  while (count < max_attempts): 
    for i in range(0, neighbourhood_size):
      candidate, cand_makespan = stochastic_2_opt(ET, city_tour = solution)
    if cand_makespan < sol_makespan:
      solution  = copy.deepcopy(candidate)
      count = 0
    else:
      count = count + 1                             
  return solution 

    # Function: Variable Neighborhood Search
def variable_neighborhood_search(ET, city_tour, max_attempts = 5, neighbourhood_size = 5, iterations = 50):
  count = 0
  solution = copy.deepcopy(city_tour)
  best_solution = copy.deepcopy(city_tour)
  best_sol_makespan = get_fitness(ET, best_solution)
  while (count < iterations):
    for i in range(0, neighbourhood_size):
      for j in range(0, neighbourhood_size):
        #solution, _ = stochastic_2_opt(ET, city_tour = best_solution)
        solution = local_search(ET, city_tour = solution, max_attempts = max_attempts, neighbourhood_size = neighbourhood_size )
        sol_makespan = get_fitness(ET,solution)
        if (sol_makespan < best_sol_makespan):
          best_solution = copy.deepcopy(solution) 
          best_sol_makespan = get_fitness(ET,best_solution)
          break
    count = count + 1
  return best_solution, best_sol_makespan