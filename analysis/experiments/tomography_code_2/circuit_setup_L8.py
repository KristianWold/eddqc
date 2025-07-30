import numpy as np
import qiskit as qk
from src import *

if __name__ == "__main__":
    np.random.seed(42)
    n = 4
    L = 8
    circuit_target = pqc_basic(n, L)
    
    circuit_SPAM_list = prepare_SPAM_strings(n)
    
    config_list, circuit_map_list = generate_circuit_sandwich(n = n, 
                                                           circuit_target = circuit_target, 
                                                           N = 5000 - 6**n)
    

    #5000 circuits in total, first 1296 are required to retrieve SPAM
    # please run all circuits at shots = 1024 and gather measured probabilities.
    circuit_list = circuit_SPAM_list + circuit_map_list
