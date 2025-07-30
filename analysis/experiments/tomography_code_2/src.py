import numpy as np
import qiskit as qk
from qiskit.quantum_info import Operator


def numberToBase(n, b, num_digits):
    """Convert a number to a given base with a fixed number of digits.
       This is used to translate an index to a configuration for the Pauli string."""
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b

    while len(digits) < num_digits:
        digits.append(0)
    return digits[::-1]



def prepare_input_string(config, return_mode="circuit"):
    """Prepare a circuit that prepares a given input Pauli string.
    0 = |0>, 1 = |1>, 2 = |+>, 3 = |->, 4 = |i+>, 5 = |i->"""
    n = len(config)
    circuit = qk.QuantumCircuit(n)
    for i, index in enumerate(config):
        if index == 0:
            pass
        if index == 1:
            circuit.rx(np.pi, i)
        if index == 2:
            circuit.ry(np.pi / 2, i)
        if index == 3:
            circuit.ry(-np.pi / 2, i)
        if index == 4:
            circuit.rx(-np.pi / 2, i)
        if index == 5:
            circuit.rx(np.pi / 2, i)


    if return_mode == "unitary":
        return Operator(circuit).data

    if return_mode == "circuit":
        return  circuit


def prepare_output_string(config, return_mode="circuit"):
    """Prepare a circuit that measures a given output Pauli string.
    0 = X, 1 = Y, 2 = Z"""
    n = len(config)

    q_reg = qk.QuantumRegister(n)
    circuit = qk.QuantumCircuit(q_reg)

    for i, index in enumerate(config):
        if index == 0:
            circuit.ry(-np.pi / 2, i)

        if index == 1:
            circuit.rx(np.pi / 2, i)

        if index == 2:
            pass  # measure in computational basis

    if return_mode == "circuit":
        return circuit

    if return_mode == "unitary":
        return Operator(circuit).data
    

def index_generator(n, N=None):
    """Generate a list of indices for the state preparation and measurement circuits."""
    index_list1 = np.arange(0, 6**n)
    index_list2 = np.arange(0, 3**n)

    
    if N is None:
        # If N is not specified, generate all possible combinations
        N = len(index_list1) * len(index_list2)

    index_list1, index_list2 = np.meshgrid(index_list1, index_list2)
    index_list = np.vstack([index_list1.flatten(), index_list2.flatten()]).T
    np.random.shuffle(index_list) # shuffle the list of indices

    # return the first N indices of the shuffled list
    return index_list[:N, 0], index_list[:N, 1]


def generate_circuit_sandwich(n=None,
                               circuit_target=None,
                               N=None,
):
    

    input_index_list, output_index_list = index_generator(n, N)


    config_list = [[], []]
    circuit_list = []
    for i, j in zip(input_index_list, output_index_list):
        config1 = numberToBase(i, 6, n)
        config2 = numberToBase(j, 3, n)

        config_list[0].append(config1)
        config_list[1].append(config2)

        state_circuit = prepare_input_string(
            config1, return_mode="circuit"
        )
        observable_circuit = prepare_output_string(
            config2, return_mode="circuit"
        )

        circuit = state_circuit
        circuit.barrier()
        circuit = circuit.compose(circuit_target)
        circuit.barrier()
        circuit = circuit.compose(observable_circuit)
        circuit.barrier()

        circuit.add_register(qk.ClassicalRegister(n))
        circuit.measure(range(n), range(n))

        circuit_list.append(circuit)

    return config_list, circuit_list


def prepare_SPAM_strings(n):

    circuit_list = []
    for i in range(6**n):
        config = numberToBase(i, 6, n)
        
        circuit = prepare_input_string(config, return_mode="circuit")
        circuit.barrier()
        circuit.add_register(qk.ClassicalRegister(n))
        circuit.measure(range(n), range(n))
        circuit_list.append(circuit)

    return circuit_list


def pqc_basic(n, L):
    """Generate a basic parameterized quantum circuit, for n qubits and L layers."""
    theta_list = [np.random.uniform(-np.pi, np.pi, 2 * n) for i in range(L)]
    circuit = qk.QuantumCircuit(n)
    for theta in theta_list:
        for i in range(n):
            circuit.ry(theta[i], i)
            circuit.rz(theta[i + n], i)

        for i in range(n - 1):
            circuit.cx(i, i + 1)

    return circuit





