import numpy as np
import scipy as sp
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import tensorflow as tf
from qiskit.quantum_info import DensityMatrix, Operator, random_unitary, Statevector
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm
from scipy.linalg import sqrtm

from utils import *
from set_precision import *


def diagonalize(M):
    eig, vec_L, vec_R = sp.linalg.eig(M, left=True, right=True)
    vec_L = vec_L.T
    vec_R = vec_R.T

    # norm between left and right eigenvectors
    norm = np.sqrt(np.einsum("ij,ij->i", vec_R, vec_L.conj()))
    vec_L /= norm[:, None].conj()
    vec_R /= norm[:, None]

    return eig, vec_L, vec_R


def partial_trace(state, discard_first=True):
    d = int(np.sqrt(state.shape[1]))
    if len(state.shape) == 2:
        state = tf.reshape(state, (d, d, d, d))
    if len(state.shape) == 3:
        state = tf.reshape(state, (-1, d, d, d, d))

    if discard_first:
        state = tf.einsum("...ijik->...jk", state)
    else:
        state = tf.einsum("...jiki->...jk", state)
    return state


def partial_trace_any(state, idx):
    d = int(state.shape[1])
    n = int(np.log2(d))
    shape = 2 * n * [2]
    state = tf.reshape(state, shape)

    state_zero = tf.gather(state, 0, axis=idx)
    state_zero = tf.gather(state_zero, 0, axis=idx + n - 1)

    state_one = tf.gather(state, 1, axis=idx)
    state_one = tf.gather(state_one, 1, axis=idx + n - 1)

    state = state_zero + state_one
    state = tf.reshape(state, (d // 2, d // 2))

    return state


def trace_superoperator(SO, indicies):
    choi = reshuffle(SO)
    n = int(np.log2(choi.shape[1])) // 2
    indicies = indicies + [i + n for i in indicies]
    for idx in reversed(indicies):
        choi = partial_trace_any(choi, idx)

    SO = reshuffle(choi / 2)
    return SO


def partial_transpose(state, qubit):
    d = state.shape[1]
    n = int(np.log2(d))
    shape = 2 * n * [2]
    state = tf.reshape(state, shape)
    new_shape = list(range(2 * n))
    new_shape[qubit] = qubit + n
    new_shape[qubit + n] = qubit
    state = tf.transpose(state, perm=new_shape)
    state = tf.reshape(state, (1, d, d))

    return state


def state_fidelity(A, B):
    sqrtB = sqrtm(B)
    C = sqrtB @ A @ sqrtB

    sqrtC = np.array(sqrtm(C), dtype=np.complex128)
    fidelity = tf.linalg.trace(sqrtC)
    return tf.abs(fidelity) ** 2


def expectation_value(probs, observable):
    ev = tf.abs(tf.reduce_sum(probs * observable, axis=1))
    return ev


# @profile
def generate_ginibre(dim1, dim2, trainable=False, complex=True):
    A = tf.random.normal((dim1, dim2), 0, 1, dtype=tf.float64)

    if complex:
        B = tf.random.normal((dim1, dim2), 0, 1, dtype=tf.float64)
    else:
        B = None
    if trainable:
        A = tf.Variable(A, trainable=True)
        if B is not None:
            B = tf.Variable(B, trainable=True)

    X = A
    if complex:
        X = tf.cast(X, dtype=precision) + 1j * tf.cast(B, dtype=precision)
    return X, A, B


def generate_state(d, rank):
    X, _, _ = generate_ginibre(d, rank)

    XX = tf.linalg.matmul(X, X, adjoint_b=True)
    state = XX / tf.linalg.trace(XX)
    return state


def generate_unitary(d=None, G=None):
    if G is None:
        G, _, _ = generate_ginibre(d, d)
    Q, R = tf.linalg.qr(G, full_matrices=False)
    D = tf.linalg.tensor_diag_part(R)
    D = tf.math.sign(D)
    D = tf.linalg.diag(D)
    U = Q @ D

    return U


def circuit_to_matrix(circuit_target, num_columns=None):
    circuit_target = circuit_target.copy().reverse_bits()
    if num_columns is None:
        U = Operator(circuit_target).data
        U = tf.convert_to_tensor(U, dtype=precision)
    else:
        n = circuit_target.num_qubits
        column_list = []
        for i in range(num_columns):
            binary = numberToBase(i, 2, n)
            circuit_basis = qk.QuantumCircuit(n)
            for j, q in enumerate(binary):
                if q:
                    circuit_basis.x(j)

            circuit = circuit_basis.compose(circuit_target)
            column = Statevector.from_instruction(circuit).data
            column_list.append(column)

        U = np.array(column_list).T
    return U


def channel_to_choi(channel_list):
    if not isinstance(channel_list, list):
        channel_list = [channel_list]

    d = channel_list[0].d
    choi = tf.zeros((d**2, d**2), dtype=precision)
    M = np.zeros((d**2, d, d))
    for i in range(d):
        for j in range(d):
            M[d * i + j, i, j] = 1

    M = tf.convert_to_tensor(M, dtype=precision)
    M_prime = tf.identity(M)
    for channel in channel_list:
        M_prime = channel.apply_channel(M_prime)

    for i in range(d**2):
        choi += tf.experimental.numpy.kron(M_prime[i], M[i])

    return choi


def apply_unitary(state, U):
    Ustate = tf.matmul(U, state)
    UstateU = tf.matmul(Ustate, U, adjoint_b=True)
    return UstateU


def attraction(channel, N=1000):
    d = channel.d
    I = tf.cast(tf.eye(d, batch_shape=(N,)), dtype=precision) / d

    state_list = []
    state = np.zeros((d, d))
    state[0, 0] = 1
    for i in range(N):
        U = random_unitary(d).data
        state_haar = DensityMatrix(U @ state @ U.T.conj()).data
        state_list.append(state_haar)

    state = tf.convert_to_tensor(state_list)

    state = channel.apply_channel(state)
    att = tf.math.reduce_mean(state_fidelity(state, I))

    return att


def corr_mat_to_povm(corr_mat):
    d = corr_mat.shape[0]
    povm = []
    for i in range(d):
        M = tf.linalg.diag(corr_mat[i, :])
        povm.append(M)

    povm = tf.convert_to_tensor(povm, dtype=precision)

    return povm


def init_ideal(d):
    init = np.zeros((d, d))
    init[0, 0] = 1
    init = tf.convert_to_tensor(init, dtype=precision)
    return init


def povm_ideal(d):
    povm = tf.cast(corr_mat_to_povm(np.eye(d)), dtype=precision)
    return povm


def measurement(state, U_basis=None, povm=None):
    d = state.shape[1]

    if povm is None:
        povm = corr_mat_to_povm(np.eye(d))

    if U_basis is not None:
        state = apply_unitary(state, U_basis)

    state = tf.expand_dims(state, axis=1)
    povm = tf.expand_dims(povm, axis=0)

    probs = tf.linalg.trace(state @ povm)

    return probs


def add_noise_to_probs(tensor, noise=0.01):
    tensor = tensor + tf.math.sqrt(tensor * (1 - tensor)) * tf.cast(
        tf.random.normal(tensor.shape, 0, noise), dtype=precision
    )
    tensor = tensor / tf.math.reduce_sum(tensor, axis=1, keepdims=True)

    return tensor


def resample(probs, shots):
    probs = tf.cast(probs, dtype=tf.float64)
    samples = tf.random.categorical(tf.math.log(probs), shots)
    samples = tf.one_hot(samples, probs.shape[1], dtype=precision)
    probs_resample = tf.math.reduce_mean(samples, axis=1)
    return probs_resample


def generate_haar_random(d, rng=np.random.default_rng()):
    U = tf.cast(Operator(random_unitary(d, seed=rng).data), dtype=precision)
    return U


def reshuffle(A):
    d = int(np.sqrt(A.shape[0]))
    A = tf.reshape(A, (d, d, d, d))
    A = tf.einsum("jklm -> jlkm", A)
    A = tf.reshape(A, (d**2, d**2))

    return A


def random_hamiltonian(d):
    A = tf.random.normal((d, d), 0, 1, dtype=tf.float64)
    B = tf.random.normal((d, d), 0, 1, dtype=tf.float64)
    G = tf.complex(A, B)
    H = (G + tf.linalg.adjoint(G)) / 2
    return H
