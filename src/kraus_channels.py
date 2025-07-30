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

from quantum_tools import *
from spam import *
from utils import *
from set_precision import *
from quantum_channel import *


def isomery_to_kraus(V, rank=None):
    rd = V.shape[0]
    if rank is None:
        d = V.shape[1]
        rank = rd // d
    else:
        d = rd // rank

    kraus = [V[i * d : (i + 1) * d, :d] for i in range(rank)]
    kraus = tf.convert_to_tensor(kraus, dtype=precision)
    kraus = tf.reshape(kraus, (1, rank, d, d))
    kraus_map = KrausMap(d, rank, trainable=False, generate=False)
    kraus_map.kraus = kraus

    return kraus_map


def kraus_marginalize(kraus_map):

    kraus_tensor = kraus_map.kraus[0]

    rank = kraus_tensor.shape[0]
    d = kraus_tensor.shape[1]
    kraus_list = []

    for i in range(rank):
        for j in range(2):
            kraus = kraus_tensor[i, j * (d // 2) : (j + 1) * (d // 2), : (d // 2)]
            kraus = tf.expand_dims(kraus, axis=0)
            kraus_list.append(kraus)

    kraus = tf.concat(kraus_list, axis=0)
    kraus = tf.expand_dims(kraus, axis=0)
    kraus_map = KrausMap(d // 2, 2 * rank, trainable=False, generate=False)
    kraus_map.kraus = kraus

    return kraus_map


class UnitaryMap(Channel):
    def __init__(
        self,
        d=None,
        spam=None,
        trainable=True,
        generate=True,
    ):
        self.d = d

        if spam is None:
            spam = SPAM(d=d, init=init_ideal(d), povm=povm_ideal(d))
        self.spam = spam

        _, self.A, self.B = generate_ginibre(d, d, trainable=trainable)
        self.parameter_list = [self.A, self.B]

        self.kraus = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = tf.cast(self.A, dtype=precision) + 1j * tf.cast(self.B, dtype=precision)
        self.U = tf.reshape(generate_unitary(G=G), (1, 1, self.d, self.d))

    def apply_channel(self, state):
        U = tf.reshape(self.U, (1, self.d, self.d))
        Ustate = tf.matmul(self.U, state)
        UstateU = tf.matmul(Ustate, self.U, adjoint_b=True)
        state = tf.reduce_sum(UstateU, axis=1)

        return state

    @property
    def choi(self):
        choi = tf.experimental.numpy.kron(self.U, tf.math.conj(self.U))
        return choi


class KrausMap(Channel):
    def __init__(
        self,
        d=None,
        rank=None,
        spam=None,
        trainable=True,
        generate=True,
    ):
        self.d = d
        self.rank = rank
        self.trainable = trainable

        if spam is None:
            spam = IdealSPAM(d=d)
        self.spam = spam

        _, self.A, self.B = generate_ginibre(rank * d, d, trainable=trainable)

        self.parameter_list = []
        if self.trainable:
            self.parameter_list = [self.A, self.B]

        self.kraus = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = tf.complex(self.A, self.B)
        U = generate_unitary(G=G)
        self.kraus = tf.reshape(U[:, : self.d], (1, self.rank, self.d, self.d))

    def apply_channel(self, state):
        state = tf.expand_dims(state, axis=1)
        Kstate = tf.matmul(self.kraus, state)
        KstateK = tf.matmul(Kstate, self.kraus, adjoint_b=True)
        state = tf.reduce_sum(KstateK, axis=1)

        return state

    @property
    def superoperator(self):
        return kraus_to_superoperator(self)

    @property
    def choi(self):
        return kraus_to_choi(self)


class DilutedKrausMap(KrausMap):
    def __init__(
        self,
        U=None,
        c=None,
        kraus_part=None,
        spam=None,
        trainable=True,
        generate=True,
    ):
        self.d = U.shape[0]
        self.rank = kraus_part.rank
        self.spam = spam
        self.kraus_part = kraus_part

        if spam is None:
            spam = IdealSPAM(d=self.d)
        self.spam = spam

        self.parameter_list = self.kraus_part.parameter_list

        self.U = U
        if self.U is not None:
            self.U = tf.expand_dims(tf.expand_dims(self.U, 0), 0)
            k = -np.log(1 / c - 1)
            self.k = tf.Variable(k, trainable=True)
            self.parameter_list.append(self.k)
        else:
            self.k = None

        self.kraus = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        if self.kraus_part.trainable:
            self.kraus_part.generate_channel()

        if self.U is not None:
            c = 1 / (1 + tf.exp(-self.k))
            c = tf.cast(c, dtype=precision)
            self.kraus = tf.concat(
                [tf.sqrt(c) * self.U, tf.sqrt(1 - c) * self.kraus_part.kraus], axis=1
            )

    @property
    def c(self):
        if self.k is None:
            c = None
        else:
            c = 1 / (1 + tf.exp(-self.k))
        return c

    @property
    def choi(self):
        return kraus_to_choi(self)


class DilutedKrausMapExtended(KrausMap):
    def __init__(
        self,
        U=None,
        d=None,
        p=None,
        rank=None,
        eps=None,
        spam=None,
        generate=True,
    ):
        self.d = d
        self.p = (p,)
        self.rank = rank
        self.eps = eps

        self.U = U

        self.kraus_part = KrausMap(d=d, rank=rank, trainable=False)
        self.unitary_noise = HermitianChannel(d=d, eps=eps, trainable=False)

        if spam is None:
            spam = IdealSPAM(d=self.d)
        self.spam = spam

        k = -np.log(p / (1 - p))
        self.k = tf.Variable(k, trainable=True)
        self.parameter_list.append(self.k)

        self.parameter_list = self.kraus_part.parameter_list

        if generate:
            self.generate_channel()

    def generate_channel(self):
        self.kraus_part.generate_channel()
        self.unitary_noise.generate_channel()

        U = self.U @ self.unitary_noise.U
        U = tf.expand_dims(tf.expand_dims(U, 0), 0)

        p = tf.exp(-self.k) / (1 + tf.exp(-self.k))
        p = tf.cast(p, dtype=precision)
        self.kraus = tf.concat(
            [tf.sqrt(p) * U, tf.sqrt(1 - p) * self.kraus_part.kraus], axis=1
        )

    @property
    def p(self):
        if self.k is None:
            p = None
        else:
            p = 1 / (1 + tf.exp(-self.k))
        return p

    @property
    def choi(self):
        return kraus_to_choi(self)


class OneLocalKrausMap(KrausMap):
    def __init__(
        self,
        d=None,
        rank=None,
        spam=None,
        trainable=True,
        generate=True,
    ):
        self.d = d
        self.rank = rank
        self.n = int(np.log2(d))
        self.I = tf.eye(2, dtype=precision)
        self.kraus_list = [KrausMap(2, rank) for i in range(self.n)]
        self.parameter_list = []
        for kraus in self.kraus_list:
            self.parameter_list.extend(kraus.parameter_list)

        self.kraus = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        self.kraus = []
        for i, kraus in enumerate(self.kraus_list):
            kraus.generate_channel()
            I_start = i * [self.I]
            I_end = (self.n - i - 1) * [self.I]
            operators = I_start + [kraus.kraus] + I_end
            self.kraus.append(kron(*operators) / np.sqrt(self.n))

        self.kraus = tf.concat(self.kraus, axis=1)


class TwoLocalKrausMap(KrausMap):
    def __init__(
        self,
        d=None,
        rank=None,
        spam=None,
        trainable=True,
        generate=True,
        start=0,
    ):
        self.d = d
        self.rank = rank
        self.n = int(np.log2(d))
        self.start = start

        self.I = tf.eye(2, dtype=precision)
        self.kraus_list = [KrausMap(4, rank) for i in range((self.n - start) // 2)]
        self.parameter_list = []
        for kraus in self.kraus_list:
            self.parameter_list.extend(kraus.parameter_list)

        self.kraus = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        self.kraus = []
        for i, kraus in enumerate(self.kraus_list):
            kraus.generate_channel()
            I_start = (i + self.start) * [self.I]
            I_end = (self.n - i - self.start - 2) * [self.I]
            operators = I_start + [kraus.kraus] + I_end
            self.kraus.append(kron(*operators) / np.sqrt(len(self.kraus_list)))

        self.kraus = tf.concat(self.kraus, axis=1)


class HermitianChannel(Channel):

    def __init__(self, d=None, eps=None, spam=None, trainable=True, generate=True):
        self.d = d
        self.eps = eps
        self.spam = spam

        if spam is None:
            spam = IdealSPAM(d=self.d)
        self.spam = spam

        _, self.A, self.B = generate_ginibre(self.d, self.d, trainable=trainable)
        self.parameter_list = [self.A, self.B]

        self.kraus = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = tf.complex(self.A, self.B)
        H = (G + tf.transpose(tf.math.conj(G))) / np.sqrt(2 * self.d)
        self.U = tf.linalg.expm(-1j * self.eps * H)
        self.kraus = isomery_to_kraus(self.U)

    def apply_channel(self, state):
        return self.kraus.apply_channel(state)
