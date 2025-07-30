import numpy as np
import numpy as np
import qiskit as qk
from src import *

if __name__ == "__main__":
    
    n = 4        # number of qubits that are measured
    n_anc = 1    # number of ancilla qubits, that are trace out/discarded/not measured
    L = 30       # layers of circuit
    reps = 10    # repeated experiments/ different target circuits

    circuit_list = []

    # same experiment repeated 10 times with different seeds
    for i in range(reps):
        np.random.seed(42 + i) # different seeds for each realization
        circuit_target = integrable_circuit(n + n_anc, L)
        
        circuit_SPAM_list = prepare_SPAM_strings(n, n_anc)
        
        config_list, circuit_map_list = generate_circuit_sandwich(n = n, 
                                                                  n_anc = n_anc,
                                                                  circuit_target = circuit_target, 
                                                                  N = 5000 - 6**n)
        

        # 5000 circuit per target circuit, 1296 for spam and 3704 for tomography
        # ten different realizations of target circuits, ing in total 50000 circuits
        # please run with shots = 1024, no compiling optimization if possible
        circuit_list.extend(circuit_SPAM_list)
        circuit_list.extend(circuit_map_list)
