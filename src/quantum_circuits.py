import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import tensorflow as tf
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator, random_unitary
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm
from qiskit.circuit.library import iSwapGate, XGate, YGate, HGate, CXGate, RGate, RZGate


def pqc_basic(n, L):
    theta_list = [np.random.uniform(-np.pi, np.pi, 2 * n) for i in range(L)]
    circuit = qk.QuantumCircuit(n)
    for theta in theta_list:
        for i in range(n):
            circuit.ry(theta[i], i)
            circuit.rz(theta[i + n], i)

        for i in range(n - 1):
            circuit.cx(i, i + 1)

    return circuit


def pqc_expressive(n, L):
    theta_list = [np.random.uniform(0, 2 * np.pi, 4 * n) for i in range(L)]
    circuit = qk.QuantumCircuit(n)
    for theta in theta_list:
        for i in range(n):
            circuit.ry(theta[i], i)

        for i in range(n):
            circuit.crx(theta[i + n], i, (i + 1) % n)

        for i in range(n):
            circuit.ry(theta[i + 2 * n], i)

        for i in reversed(list(range(n))):
            circuit.crx(theta[3 * n + i], (i + 1) % n, i)

    return circuit


def pqc_more_expressive(n, L):
    theta_list = [np.random.uniform(-np.pi, np.pi, 4 * n) for i in range(L)]
    circuit = qk.QuantumCircuit(n)
    for theta in theta_list:
        for i in range(n):
            circuit.ry(theta[i], i)
            circuit.rz(theta[i + n], i)

        for i in range(n):
            circuit.cx(i, (i + 1) % n)

        for i in range(n):
            circuit.ry(theta[i + 2 * n], i)
            circuit.rx(theta[i + 3 * n], i)

        for i in range(n):
            circuit.cx(n - i - 1, n - (i + 1) % n - 1)

    return circuit


def integrable_circuit(n, L):
    theta_list = [np.random.uniform(-np.pi, np.pi, n) for i in range(L)]
    sqrt_iSWAP = iSwapGate().power(1 / 2)

    circuit = qk.QuantumCircuit(n)
    for i, theta in enumerate(theta_list):

        offset = i % 2
        for j in range(n):
            circuit.rz(theta[j], j)

        for j in range((n - offset) // 2):
            circuit.append(sqrt_iSWAP, [2 * j + offset, 2 * j + 1 + offset])

    return circuit


def nonintegrable_circuit(n, L):
    theta_list = [np.random.uniform(-np.pi, np.pi, 2 * n) for i in range(L)]
    sqrt_iSWAP = iSwapGate().power(1 / 2)

    circuit = qk.QuantumCircuit(n)
    for i, theta in enumerate(theta_list):

        offset = i % 2
        for j in range(n):
            circuit.ry(theta[j], j)
            circuit.rz(theta[j + n], j)

        for j in range((n - offset) // 2):
            circuit.append(sqrt_iSWAP, [2 * j + offset, 2 * j + 1 + offset])

    return circuit

def haar_random(n, L):
    circuit = random_unitary(2**n)

    return circuit



"""
def integrable_circuit(n, L, use_hadamard=False):
    theta_list = [np.random.uniform(-np.pi, np.pi, 2 * n) for i in range(L)]
    sqrt_iSWAP = iSwapGate().power(1 / 2)

    circuit = qk.QuantumCircuit(n)
    if use_hadamard:
        for i in range(n):
            circuit.h(i)
    for theta in theta_list:
        for i in range(n):
            circuit.rz(theta[i], i)

        for i in range(n // 2):
            circuit.append(sqrt_iSWAP, [2 * i, 2 * i + 1])

        for i in range(n):
            circuit.rz(theta[n + i], i)

        for i in range((n - 1) // 2):
            circuit.append(sqrt_iSWAP, [2 * i + 1, 2 * i + 2])

    return circuit


def integrable_circuit_alt(n, L, use_hadamard=False):
    theta_list = [np.random.uniform(-np.pi, np.pi, 3 * n) for i in range(L)]
    sqrt_iSWAP = iSwapGate().power(1 / 2)

    circuit = qk.QuantumCircuit(n)
    if use_hadamard:
        for i in range(n):
            circuit.h(i)

    for theta in theta_list:
        for i in range(n):
            circuit.rz(theta[i], i)

        for i in range(n // 2):
            circuit.append(sqrt_iSWAP, [2 * i, 2 * i + 1])

        for i in range(n):
            circuit.rz(theta[n + i], i)

        for i in range((n - 1) // 2):
            circuit.append(sqrt_iSWAP, [2 * i + 1, 2 * i + 2])

        for i in range(n):
            circuit.rz(theta[2 * n + i], i)

        for i in range(n // 2):
            circuit.append(sqrt_iSWAP, [2 * i, 2 * i + 1])

    return circuit


def integrable_circuit_star(n, L):
    theta_list = [np.random.uniform(-np.pi, np.pi, 8) for i in range(L)]
    sqrt_iSWAP = iSwapGate().power(1 / 2)

    circuit = qk.QuantumCircuit(n)

    for theta in theta_list:
        circuit.rz(theta[0], 0)
        circuit.rz(theta[1], 2)
        circuit.append(sqrt_iSWAP, [0, 2])

        circuit.rz(theta[2], 1)
        circuit.rz(theta[3], 2)
        circuit.append(sqrt_iSWAP, [1, 2])

        circuit.rz(theta[4], 3)
        circuit.rz(theta[5], 2)
        circuit.append(sqrt_iSWAP, [3, 2])

        if n > 4:
            circuit.rz(theta[6], 4)
            circuit.rz(theta[7], 2)
            circuit.append(sqrt_iSWAP, [4, 2])

    return circuit


def nonintegrable_circuit_star(n, L):
    index_list = [np.random.randint(0, 8, 8) for i in range(L)]
    sqrt_iSWAP = iSwapGate().power(1 / 2)

    gate_list = [
        XGate().power(1 / 2),
        XGate().power(-1 / 2),
        YGate().power(1 / 2),
        YGate().power(-1 / 2),
        RGate(np.pi / 2, np.pi / 4),
        RGate(-np.pi / 2, np.pi / 4),
        RGate(np.pi / 2, -np.pi / 4),
        RGate(-np.pi / 2, -np.pi / 4),
    ]

    circuit = qk.QuantumCircuit(5)

    for index in index_list:
        circuit.append(gate_list[index[0]], [0])
        circuit.append(gate_list[index[1]], [2])
        circuit.append(sqrt_iSWAP, [0, 2])

        circuit.append(gate_list[index[2]], [1])
        circuit.append(gate_list[index[3]], [2])
        circuit.append(sqrt_iSWAP, [1, 2])

        circuit.append(gate_list[index[4]], [3])
        circuit.append(gate_list[index[5]], [2])
        circuit.append(sqrt_iSWAP, [3, 2])

        circuit.append(gate_list[index[6]], [4])
        circuit.append(gate_list[index[7]], [2])
        circuit.append(sqrt_iSWAP, [4, 2])

    return circuit


def nonintegrable_circuit(n, L, use_hadamard=False, use_sqrtSwap=True):
    # index_list = [np.random.randint(0, 8, 3 * n) for i in range(L)]
    index_list = [np.random.randint(0, 8, 2 * n) for i in range(L)]
    gate_list = [
        XGate().power(1 / 2),
        XGate().power(-1 / 2),
        YGate().power(1 / 2),
        YGate().power(-1 / 2),
        RGate(np.pi / 2, np.pi / 4),
        RGate(-np.pi / 2, np.pi / 4),
        RGate(np.pi / 2, -np.pi / 4),
        RGate(-np.pi / 2, -np.pi / 4),
    ]
    if use_sqrtSwap:
        ent_gate = iSwapGate().power(1 / 2)
    else:
        ent_gate = iSwapGate()

    circuit = qk.QuantumCircuit(n)
    if use_hadamard:
        for i in range(n):
            circuit.h(i)

    for index in index_list:
        for i in range(n):
            circuit.append(gate_list[index[i]], [i])

        for i in range(n // 2):
            circuit.append(ent_gate, [2 * i, 2 * i + 1])

        for i in range(n):
            circuit.append(gate_list[index[n + i]], [i])

        for i in range((n - 1) // 2):
            circuit.append(ent_gate, [2 * i + 1, 2 * i + 2])

        # for i in range(n):
        #    circuit.append(gate_list[index[2 * n + i]], [i])

        # for i in range(n // 2):
        #    circuit.append(ent_gate, [2 * i, 2 * i + 1])

    return circuit


def clifford_circuit(n, L, use_hadamard=False):
    index_list = [np.random.randint(0, 3, 2 * n) for i in range(L)]
    gate_list = [XGate(), YGate(), HGate()]
    ent_gate = CXGate()

    circuit = qk.QuantumCircuit(n)
    if use_hadamard:
        for i in range(n):
            circuit.h(i)

    for index in index_list:
        for i in range(n):
            circuit.append(gate_list[index[i]], [i])

        for i in range(n // 2):
            circuit.append(ent_gate, [2 * i, 2 * i + 1])

        for i in range(n):
            circuit.append(gate_list[index[n + i]], [i])

        for i in range((n - 1) // 2):
            circuit.append(ent_gate, [2 * i + 1, 2 * i + 2])

    return circuit
"""
