import argparse, sys
import os
import time
import numpy as np
from numba import jit
from utils import get_initial_pop, get_big_pop, mutate, get_fitness, order_pop
from utils import create_prob_matrix, fill_prob_matrix, create_new_pop, add_heuristics
from utils import save_to_csv, variable_neighborhood_search


parser=argparse.ArgumentParser()

parser.add_argument('--jobs', help='number of jobs')
parser.add_argument('--machines', help='number of jobs to schedule')
parser.add_argument('--path', help='path for the jobs instance')
parser.add_argument('--numInd', help='number of individuals')
parser.add_argument('--numGen', help='number of generations')
parser.add_argument('--toMatrix', help='percentual da matriz')
parser.add_argument('--elitism', help='percentual de individuos que passam de Geração')
parser.add_argument('--mutation', help='percentual de individuos que sofrem mutacao')



args=parser.parse_args()

jobs = [int(i) for i in args.jobs.split(',')]
machines = [int(i) for i in args.machines.split(',')]
numInd = [int(i) for i in args.numInd.split(',')]
numGen = [int(i) for i in args.numGen.split(',')]
toMatrix = [float(i) for i in args.toMatrix.split(',')]
path = [i for i in args.path.split(',')]
elitism = [int(i) for i in args.elitism.split(',')]
mutation = [int(i) for i in args.mutation.split(',')]

x = np.arange(100).reshape(10, 10)

@jit(nopython=True)
def go_fast(a): # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


for j in jobs:
    for m in machines:
        for ni in numInd:
            for ng in numGen:
                for tm in toMatrix:
                    for p in path:
                        for e in elitism:
                            for mu in mutation:
                                array = np.array(open(p).readlines(),dtype=float)
                                ET = np.reshape(array,(j, m))
                                CT = ET.copy()
                                
                                start = time.time()

                                initial_pop = get_initial_pop()
                                big_pop = get_big_pop(p)

                                fst_pop = np.append([initial_pop], [big_pop], axis=1)[0]

                                mutate(fst_pop, mu, m)

                                heuristics = add_heuristics(ET, CT, m)
                                fst_pop = np.append([fst_pop], [heuristics], axis=1)[0]
                                
                                fst_pop = order_pop(ET, fst_pop)
                                fst_pop = fst_pop[:ni]

                                best_makespan = get_fitness(ET, fst_pop[-1])

                                for i in range(ng):
                                    #print(f'Gen {i}...')
                                    mutate(fst_pop, mu, m)
                                    pb = create_prob_matrix(j, m)
                                    fst_pop = order_pop(ET, fst_pop)

                                    pb_to_matrix = fst_pop.copy()[:int(len(fst_pop)*tm)]
                                    fill_prob_matrix(pb, pb_to_matrix)
                                    
                                    fst_pop = fst_pop[:e]
                                    
                                    new_pop = create_new_pop(pb, ni - len(fst_pop))
                                    fst_pop = np.append([fst_pop], [new_pop], axis=1)[0]
                                    
                                    fst_pop = order_pop(ET, fst_pop)

                                    if i == ng - 1:
                                        individual, fitness = variable_neighborhood_search(ET, fst_pop[-1])
                                        fst_pop[-1] = individual

                                    if get_fitness(ET, fst_pop[-1]) < best_makespan:
                                        best_makespan = get_fitness(ET, fst_pop[-1])

                                end = time.time()
                                elapsed_time = end - start
                                
                                print("Elapsed time = %s" % (elapsed_time))
                                print(f"Best_Makespan for {p} is {best_makespan}...")
                                save_to_csv(j, m, ni, ng, best_makespan, tm, elapsed_time, 'roulette', e, p, mu)