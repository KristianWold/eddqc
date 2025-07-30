import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import tensorflow as tf
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm
from copy import deepcopy
from set_precision import *

from quantum_tools import *
from utils import *
from set_precision import *


def generate_corruption_matrix(counts_list):
    n = len(list(counts_list[0].keys())[0])
    corr_mat = np.zeros((2**n, 2**n))
    for i, counts in enumerate(counts_list):
        for string, value in counts.items():
            index = int(string, 2)
            corr_mat[index, i] = value

    corr_mat = corr_mat / sum(counts_list[0].values())
    return corr_mat


class InitialState:
    def __init__(self, d, c=None, trainable=True):
        self.d = d

        self.A = tf.random.normal((d, d), 0, 1, dtype=tf.float64)
        self.B = tf.random.normal((d, d), 0, 1, dtype=tf.float64)
        self.init_ideal = init_ideal(d)

        if trainable:
            self.A = tf.Variable(self.A, trainable=True)
            self.B = tf.Variable(self.B, trainable=True)

        self.parameter_list = [self.A, self.B]

        if c is None:
            self.k = None
        else:
            k = -np.log(1 / c - 1)
            self.k = tf.Variable(k, trainable=True)
            self.parameter_list.append(self.k)

        self.generate_init()

    def generate_init(self):
        G = tf.complex(self.A, self.B)
        AA = tf.matmul(G, G, adjoint_b=True)
        self.init = AA / tf.linalg.trace(AA)
        if self.k is not None:
            self.init = self.c * self.init_ideal + (1 - self.c) * self.init

    @property
    def c(self):
        return tf.cast(1 / (1 + tf.exp(-self.k)), dtype=precision)


class POVM:
    def __init__(self, d, c=None, trainable=True):
        self.d = d
        self.A = tf.random.normal((d, d, d), 0, 1, dtype=tf.float64)
        self.B = tf.random.normal((d, d, d), 0, 1, dtype=tf.float64)
        self.povm_ideal = povm_ideal(d)

        if trainable:
            self.A = tf.Variable(self.A, trainable=True)
            self.B = tf.Variable(self.B, trainable=True)

        self.parameter_list = [self.A, self.B]

        if c is None:
            self.k = None
        else:
            k = -np.log(1 / c - 1)
            self.k = tf.Variable(k, trainable=True)
            self.parameter_list.append(self.k)

        self.generate_POVM()

    def generate_POVM(self):
        G = tf.complex(self.A, self.B)
        AA = tf.matmul(G, G, adjoint_b=True)
        D = tf.math.reduce_sum(AA, axis=0)
        invsqrtD = tf.linalg.inv(tf.linalg.sqrtm(D))
        self.povm = tf.matmul(tf.matmul(invsqrtD, AA), invsqrtD)
        if self.k is not None:
            self.povm = self.c * self.povm_ideal + (1 - self.c) * self.povm

    @property
    def c(self):
        return tf.cast(1 / (1 + tf.exp(-self.k)), dtype=precision)



class POVMwQR:
    def __init__(self, d, c=None, trainable=True):
        self.d = d
        self.A = tf.random.normal((d**2, d), 0, 1, dtype=tf.float64)
        self.B = tf.random.normal((d**2, d), 0, 1, dtype=tf.float64)
        self.povm_ideal = povm_ideal(d)

        if trainable:
            self.A = tf.Variable(self.A, trainable=True)
            self.B = tf.Variable(self.B, trainable=True)

        self.parameter_list = [self.A, self.B]

        if c is None:
            self.k = None
        else:
            k = -np.log(1 / c - 1)
            self.k = tf.Variable(k, trainable=True)
            self.parameter_list.append(self.k)

        self.generate_POVM()

    def generate_POVM(self):
        G = tf.complex(self.A, self.B)
        U = generate_unitary(G=G)
        kraus = tf.reshape(U[:, : self.d], (self.d, self.d, self.d))
        self.povm = tf.matmul(kraus, kraus, adjoint_b=True)


        if self.k is not None:
            self.povm = self.c * self.povm_ideal + (1 - self.c) * self.povm

    @property
    def c(self):
        return tf.cast(1 / (1 + tf.exp(-self.k)), dtype=precision)


class CorruptionMatrix:
    def __init__(self, d, c=None, trainable=True):
        self.d = d
        self.A = tf.random.normal((d, d), 0, 1, dtype=tf.float64)
        self.povm_ideal = povm_ideal(d)

        if trainable:
            self.A = tf.Variable(self.A, trainable=True)

        self.parameter_list = [self.A]

        if c is None:
            self.k = None
        else:
            k = -np.log(1 / c - 1)
            self.k = tf.Variable(k, trainable=True, dtype=precision)
            self.parameter_list.append(self.k)

        self.povm = None
        self.generate_POVM()

    def generate_POVM(self):
        C = tf.abs(self.A)
        C = C / tf.reduce_sum(C, axis=0)
        self.povm = tf.cast(corr_mat_to_povm(C), dtype=precision)
        if self.k is not None:
            self.povm = self.c * self.povm_ideal + (1 - self.c) * self.povm

    @property
    def c(self):
        return tf.cast(1 / (1 + tf.exp(-self.k)), dtype=precision)


class SPAM:
    def __init__(
        self,
        init=None,
        povm=None,
    ):
        self.d = init.d
        self.init = init
        self.povm = povm

        self.parameter_list = self.init.parameter_list + self.povm.parameter_list

        self.generate_SPAM()


    def generate_SPAM(self):
        self.init.generate_init()
        self.povm.generate_POVM()


class IdealSPAM:
    def __init__(self, d):
        self.d = d
        self.init = IdealInit(d)
        self.povm = IdealPOVM(d)


class IdealInit:
    def __init__(self, d):
        self.d = d
        self.init = init_ideal(d)
        self.parameter_list = []

    def generate_init(self):
        pass


class IdealPOVM:
    def __init__(self, d):
        self.d = d
        self.povm = povm_ideal(d)
        self.parameter_list = []

    def generate_POVM(self):
        pass


def povm_fidelity(povm_a, povm_b):
    d = povm_a.shape[0]
    ab = tf.matmul(povm_a, povm_b)
    ab_sqrt = tf.linalg.sqrtm(ab)
    fidelity = tf.math.reduce_sum(tf.linalg.trace(ab_sqrt)) / d
    return tf.abs(fidelity)**2


def generate_spam_benchmark(n=3, c1=1, c2=1):
    d = 2**n

    init_target = InitialState(d, c=c1)
    povm_target = POVM(d, c=c2)

    spam_target = SPAM(init=init_target, povm=povm_target)

    return spam_target
