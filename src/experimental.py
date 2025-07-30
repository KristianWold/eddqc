import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
from qiskit.quantum_info import DensityMatrix, Operator, random_unitary
from scipy.linalg import sqrtm

# from qiskit.tools.monitor import job_monitor
from tqdm.notebook import tqdm
from qiskit import QuantumCircuit, QuantumRegister

from quantum_tools import *
from quantum_channel import *
from utils import *
from set_precision import *


# @profile
def prepare_input(config, return_mode="density", reverse=True):
    """0 = |0>, 1 = |1>, 2 = |+>, 3 = |->, 4 = |i+>, 5 = |i+>"""
    n = len(config)
    circuit = qk.QuantumCircuit(n)
    for i, gate in enumerate(config):
        if gate == 0:
            pass
        if gate == 1:
            circuit.rx(np.pi, i)
        if gate == 2:
            circuit.ry(np.pi / 2, i)
        if gate == 3:
            circuit.ry(-np.pi / 2, i)
        if gate == 4:
            circuit.rx(-np.pi / 2, i)
        if gate == 5:
            circuit.rx(np.pi / 2, i)

    if return_mode == "density":
        if reverse:
            circuit = circuit.reverse_bits()
        state = DensityMatrix(circuit).data

    if return_mode == "unitary":
        if reverse:
            circuit = circuit.reverse_bits()
        state = Operator(circuit).data

    if return_mode == "circuit":
        if reverse:
            circuit = circuit.reverse_bits()
        state = circuit

    if return_mode == "circuit_measure":
        circuit.add_register(qk.ClassicalRegister(n))
        circuit.measure(circuit.qregs[0], circuit.cregs[0])
        if reverse:
            circuit = circuit.reverse_bits()
        state = circuit

    return state


def pauli_observable(config, return_mode="density", reverse=True):

    n = len(config)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    basis = [X, Y, Z, I]

    if return_mode == "density":
        string = [basis[idx] for idx in config]
        result = kron(*string)

    q_reg = qk.QuantumRegister(n)
    c_reg = qk.ClassicalRegister(n)
    circuit = qk.QuantumCircuit(q_reg, c_reg)

    for i, index in enumerate(config):
        if index == 0:
            circuit.ry(-np.pi / 2, i)

        if index == 1:
            circuit.rx(np.pi / 2, i)

        if index == 2:
            pass  # measure in computational basis

    if return_mode == "circuit":
        circuit.measure(q_reg, c_reg)
        if reverse:
            circuit = circuit.reverse_bits()
        result = circuit

    if return_mode == "unitary":
        if reverse:
            circuit = circuit.reverse_bits()
        result = Operator(circuit).data

    return result


def generate_pauli_circuits(
    n=None,
    circuit_target=None,
    N=None,
    trace=False,
    grid=True,
    reverse=True,
):
    state_index, observ_index = index_generator(n, N, trace=trace, grid=grid)

    if trace:
        num_observ = 4
    else:
        num_observ = 3

    input_list = [[], []]
    config_list = [[], []]
    circuit_list = []
    for i, j in zip(state_index, observ_index):
        config1 = numberToBase(i, 6, n)
        U_prep = prepare_input(config1, return_mode="unitary", reverse=reverse)

        config2 = numberToBase(j, num_observ, n)
        U_basis = pauli_observable(config2, return_mode="unitary", reverse=reverse)

        config_list[0].append(config1)
        config_list[1].append(config2)

        input_list[0].append(U_prep)
        input_list[1].append(U_basis)

        if circuit_target is not None:
            state_circuit = prepare_input(
                config1, return_mode="circuit", reverse=reverse
            )
            observable_circuit = pauli_observable(
                config2, return_mode="circuit", reverse=reverse
            )

            circuit = state_circuit
            circuit.barrier()
            circuit = circuit.compose(circuit_target)
            circuit.barrier()
            circuit.add_register(observable_circuit.cregs[0])
            circuit = circuit.compose(observable_circuit)

            circuit_list.append(circuit)

    input_list[0] = tf.convert_to_tensor(input_list[0], dtype=precision)
    input_list[1] = tf.convert_to_tensor(input_list[1], dtype=precision)

    return input_list, circuit_list


def generate_pauliInput_circuits(n=None, reverse=True):
    input_list = []
    circuit_list = []
    for i in range(6**n):
        config = numberToBase(i, 6, n)
        U_prep = prepare_input(config, return_mode="unitary", reverse=reverse)
        circuit = prepare_input(config, return_mode="circuit_measure", reverse=reverse)

        input_list.append(U_prep)
        circuit_list.append(circuit)

    input_list = tf.convert_to_tensor(input_list, dtype=precision)

    return input_list, circuit_list



def generate_pauliInput_circuits_subset(n=None, N_spam=None, reverse=True):
    input_list = []
    circuit_list = []

    index_list = np.random.choice(6**n, N_spam, replace=False)
    for i in index_list:
        config = numberToBase(i, 6, n)
        U_prep = prepare_input(config, return_mode="unitary", reverse=reverse)
        circuit = prepare_input(config, return_mode="circuit_measure", reverse=reverse)

        input_list.append(U_prep)
        circuit_list.append(circuit)

    input_list = tf.convert_to_tensor(input_list, dtype=precision)

    return input_list, circuit_list



def generate_bitstring_circuits(n):
    circuit_list = []
    for i in range(2**n):
        q_reg = qk.QuantumRegister(n)
        c_reg = qk.ClassicalRegister(n)
        circuit = qk.QuantumCircuit(q_reg, c_reg)
        config = numberToBase(i, 2, n)
        for j, index in enumerate(config):
            if index:
                circuit.x(j)
        circuit.measure(q_reg, c_reg)
        circuit_list.append(circuit.reverse_bits())

    return circuit_list


def counts_to_probs(counts_list, reversed=True):
    N = len(counts_list)
    n = len(list(counts_list[0].keys())[0])
    probs = np.zeros((N, 2**n))
    for i in range(N):
        for string, value in counts_list[i].items():
            if reversed:
                string = string[::-1]
            index = int(string, 2)
            probs[i, index] = value
    probs = tf.convert_to_tensor(probs, dtype=precision)
    probs = tf.linalg.normalize(probs, ord=1, axis=1)[0]
    return probs


def marginalize_counts(counts_list, site):
    counts_new_list = []
    for counts in counts_list:
        dict = {}
        for key, value in counts.items():
            new_key = key[:site] + key[site+1:]
            if new_key in dict:
                dict[new_key] += value
            else:
                dict[new_key] = value

        counts_new_list.append(dict)

    return counts_new_list


def generate_sandwich_circuits(target_circuit, input_circuit_list, output_circuit_list):
    circuit_list = []
    n = len(input_circuit_list[0].qregs[0])
    for i in range(len(input_circuit_list)):
        circuit = input_circuit_list[i]
        circuit.barrier()
        circuit = circuit.compose(target_circuit)
        circuit.barrier()
        circuit = circuit.compose(output_circuit_list[i])

        circuit.add_register(qk.ClassicalRegister(n))
        circuit.measure(circuit.qregs[0], circuit.cregs[0])

        circuit_list.append(circuit)

    return circuit_list


class Inputs:

    def __init__(self, n, batch_size, grid=False):
        self.n = n
        self.batch_size = batch_size
        self.index_list = index_generator(self.n, N=batch_size, grid=grid, trace=False)

        self.config_input = []
        self.config_basis = []
        for i, j in zip(*self.index_list):
            self.config_input.append(numberToBase(i, 6, self.n))
            self.config_basis.append(numberToBase(j, 3, self.n))

        self.config_input = tf.convert_to_tensor(self.config_input, dtype=tf.int32)
        self.config_basis = tf.convert_to_tensor(self.config_basis, dtype=tf.int32)
        self.generate_unitaries()

    def generate_unitaries(self):
        self.input_list = [[], []]
        for i in range(self.batch_size):
            U_prep = prepare_input(self.config_input[i], return_mode="unitary")

            U_basis = pauli_observable(self.config_basis[i], return_mode="unitary")

            self.input_list[0].append(U_prep)
            self.input_list[1].append(U_basis)

        self.input_list[0] = tf.convert_to_tensor(self.input_list[0], dtype=precision)
        self.input_list[1] = tf.convert_to_tensor(self.input_list[1], dtype=precision)

    def generate_circuits(self):
        self.circuit_list = []
        for i in range(self.batch_size):
            circuit = prepare_input(self.config_input[i], return_mode="circuit")
            self.circuit_list.append(circuit)

    def __call__(self, slice=None):
        if not slice is None:
            return [self.index_list[0][slice], self.index_list[1][slice]]
        else:
            return self.index_list

    def delete_unitaries(self):
        self.input_list = None


class Targets:

    def __init__(self, targets):
        self.targets = targets

        self.targets_original = targets
        self.targets_resample = None
        self.use_resample = False

        self.batch_size, self.d = targets.shape

    def resample(self, shots):
        self.targets_resample = resample(self.targets, shots)
        self.targets = self.targets_resample

        self.use_resample = True

    def reset_to_original(self):
        self.targets = self.targets_original
        del self.targets_resample

        self.use_resample = False

    def __call__(self, slice=None):
        if slice is None:
            return self.targets
        else:
            return self.targets[slice]


class Data:

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __call__(self, slice=None):
        return self.inputs(slice), self.targets(slice)


def marginalize_targets(targets, site):
    targets = deepcopy(targets)
    batch_size = targets.batch_size
    d = targets.d

    n = int(np.log2(d))
    new_shape = n * [2]

    targets_temp = targets.targets
    targets_temp = tf.reshape(targets_temp, shape=(-1, *new_shape))
    targets_temp = tf.math.reduce_sum(targets_temp, axis=site + 1)
    targets_temp = tf.reshape(targets_temp, shape=(batch_size, -1))

    targets.targets = targets_temp
    targets.d = targets_temp.shape[1]
    return targets


def marginalize_inputs(inputs, site):
    inputs = deepcopy(inputs)
    inputs.n -= 1
    inputs.config_input = tf.concat(
        [inputs.config_input[:, :site], inputs.config_input[:, site + 1 :]], axis=1
    )
    inputs.config_basis = tf.concat(
        [inputs.config_basis[:, :site], inputs.config_basis[:, site + 1 :]], axis=1
    )
    inputs.generate_unitaries()

    return inputs


class ExecuteHelmi:
    def setup_circuits(
        self, circuit_target, N_map=None, N_spam=None, initial_layout=None
    ):
        self.circuit_target = circuit_target

        self.n = len(circuit_target.qregs[0])

        self.inputs_spam, self.circuit_list_spam = generate_pauliInput_circuits(self.n)

        self.input_list, self.config_list, self.circuit_list_map = (
            generate_pauli_circuits(self.n, circuit_target, N=N_map)
        )

    def execute_circuits(
        self,
        backend,
        shots_map,
        shots_spam,
        batch_size,
        filename=None,
    ):
        self.shots_map = shots_map
        self.shots_spam = shots_spam

        circuit_list = self.circuit_list_spam + self.circuit_list_map
        counts_list = self.runner(
            circuit_list, batch_size=batch_size, backend=backend, shots=shots_map
        )

        counts_spam = counts_list[: len(self.circuit_list_spam)]
        counts_map = counts_list[len(self.circuit_list_spam) :]

        probs_map = counts_to_probs(counts_map)
        probs_spam = counts_to_probs(counts_spam)
        result_list = [self.input_list, probs_map, probs_spam]

        with open(filename, "wb") as handle:
            pickle.dump(result_list, handle)

    def runner(
        self,
        circuit_list,
        backend,
        shots,
        batch_size,
    ):
        N = len(circuit_list)
        num_batches = (N + batch_size - 1) // batch_size
        circuit_batch_list = [
            circuit_list[batch_size * i : batch_size * (i + 1)]
            for i in range(num_batches)
        ]
        counts_list = []
        for circuit_batch in circuit_batch_list:
            transpiled_circuit_batch = transpile(
                circuit_batch,
                backend=backend,
                layout_method="sabre",
                optimization_level=0,
                seed_transpiler=42,
            )
            job = execute(transpiled_circuit_batch, backend=backend, shots=shots)
            job_monitor(job)
            result = job.result()
            counts_list.extend(
                [result.get_counts(circuit) for circuit in circuit_batch]
            )

        return counts_list
