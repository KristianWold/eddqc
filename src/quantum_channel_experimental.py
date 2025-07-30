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

from quantum_tools import *
from spam import *
from utils import *
from set_precision import *
from spectrum import *
from quantum_channel import Channel
from kraus_channels import KrausMap, UnitaryMap

class IntegrableChoiMap(Channel):
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

        if spam is None:
            spam = IdealSPAM(self.d)
        self.spam = spam
        self.I = tf.eye(d, dtype=precision)

        _, self.A, self.B = generate_ginibre(d**2, d**2, trainable=trainable)
        self.eig = tf.Variable(tf.random.normal((d**2,), 0, 1), trainable=trainable)
        self.parameter_list = [self.A, self.B, self.eig]

        self.super_operator = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = tf.complex(self.A, self.B) / self.d
        U = generate_unitary(G=G)
        eig = tf.cast(tf.abs(self.eig), precision)
        XX = tf.linalg.diag(eig)
        XX = tf.matmul(U, XX)
        XX = tf.matmul(XX, U, adjoint_b=True)

        Y = partial_trace(XX)
        Y = tf.linalg.sqrtm(Y)
        Y = tf.linalg.inv(Y)
        Ykron = kron(self.I, Y)

        choi = Ykron @ XX @ Ykron
        self.super_operator = reshuffle(choi)

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, self.d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, self.d, self.d))

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)
    

class PTPMap:
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

        if spam is None:
            spam = SPAM(d=d, init=init_ideal(d), povm=povm_ideal(d))
        self.spam = spam

        _, self.A, self.B = generate_ginibre(d**2, rank, trainable=trainable)
        self.parameter_list = [self.A, self.B]

        self.super_operator = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = tf.complex(self.A, self.B) / self.d
        GG = tf.matmul(G, G, adjoint_b=True)
        GG = self.d * GG / tf.linalg.trace(GG)
        self.super_operator = reshuffle(GG)

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, self.d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, self.d, self.d))

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)


class PMap:
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

        if spam is None:
            spam = SPAM(d=d, init=init_ideal(d), povm=povm_ideal(d))
        self.spam = spam

        _, self.A, self.B = generate_ginibre(d**2, rank, trainable=trainable)
        self.parameter_list = [self.A, self.B]

        self.super_operator = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = tf.complex(self.A, self.B) / self.d
        GG = tf.matmul(G, G, adjoint_b=True)
        self.super_operator = reshuffle(GG)

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, self.d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, self.d, self.d))

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)
    

class LinearMap:
    def __init__(
        self,
        d=None,
        spam=None,
        trainable=True,
        generate=True,
    ):
        self.d = d


        if spam is None:
            spam = IdealSPAM(self.d)
        self.spam = spam

        _, self.A, self.B = generate_ginibre(d**2, d**2, trainable=trainable)
        self.parameter_list = [self.A, self.B]

        self.super_operator = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = tf.complex(self.A, self.B) / self.d
        self.super_operator = G

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, self.d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, self.d, self.d))

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)


class HubbardKraus(KrausMap):

    def __init__(self, d=None, spam=None, p=None, lamb=None, gamma=None):
        self.d = d
        self.n = int(np.log2(d))
        self.spam = spam

        if spam is None:
            spam = IdealSPAM(d=d)
        self.spam = spam

        self.p = tf.cast(p, dtype=precision)
        self.lamb_min, self.lamb_pos = lamb
        self.gamma_min, self.gamma_pos = gamma

        a_min = tf.math.sin(1j * self.lamb_min + self.gamma_min)
        b_min = tf.math.sin(1j * self.lamb_min)
        c_min = tf.math.sin(self.gamma_min)

        a_pos = tf.math.sin(1j * self.lamb_pos + self.gamma_pos)
        b_pos = tf.math.sin(1j * self.lamb_pos)
        c_pos = tf.math.sin(self.gamma_pos)

        R_min = (
            1
            / a_min
            * np.array(
                [
                    [a_min, 0, 0, 0],
                    [0, c_min, -1j * b_min, 0],
                    [0, 1j * b_min, c_min, 0],
                    [0, 0, 0, a_min],
                ]
            )
        )
        R_pos = (
            1
            / a_pos
            * np.array(
                [
                    [a_pos, 0, 0, 0],
                    [0, c_pos, -1j * b_pos, 0],
                    [0, 1j * b_pos, c_pos, 0],
                    [0, 0, 0, a_pos],
                ]
            )
        )

        R_min = tf.cast(R_min, dtype=precision)
        R_pos = tf.cast(R_pos, dtype=precision)

        I = tf.eye(2, dtype=precision)
        Z = tf.convert_to_tensor([[1, 0], [0, -1]], dtype=precision)

        K_min = tf.math.sqrt(1 - self.p) * R_min
        K_pos = tf.math.sqrt(self.p) * R_pos @ (kron(Z, I))
        SO = tf.eye(self.d**2, dtype=precision)
        for i in range(self.n - 1):
            pre = i * [I]
            post = (self.n - i - 2) * [I]
            K_min_dilated = kron(*(pre + [K_min] + post))
            K_pos_dilated = kron(*(pre + [K_pos] + post))
            SO_loc = kron(K_min_dilated, tf.math.conj(K_min_dilated)) + kron(
                K_pos_dilated, tf.math.conj(K_pos_dilated)
            )

            SO = SO_loc @ SO

        circuit = qk.QuantumCircuit(self.n)
        for i in range(self.n - 1):
            circuit.swap(i, i + 1)
        T = Operator(circuit).data
        T = tf.cast(T, dtype=precision)
        T = kron(T, T)
        self.SO = tf.linalg.inv(T) @ SO @ T @ SO


class EnsambleDilutedUnitary:
    def __init__(
        self,
        d=None,
        c=None,
        rank=None,
        U=None,
        samples=None,
        spam=None,
        trainable=True,
        generate=True,
    ):
        self.d = d
        self.rank = rank
        self.spam = spam
        self.samples = samples
        self.trainable = trainable

        if spam is None:
            spam = IdealSPAM(d=d)
        self.spam = spam

        k = -np.log(1 / c - 1)
        self.k = tf.Variable(k, trainable=True)

        self.parameter_list = [self.k]

        self.ensamble = []
        for _ in range(self.samples):
            if U is None:
                _U = generate_unitary(self.d)
            else:
                _U = U
            self.ensamble.append(
                DilutedKrausMap(
                    U=_U,
                    c=c,
                    kraus_part=KrausMap(d=d, rank=rank, trainable=False),
                    spam=spam,
                    trainable=False,
                )
            )

            self.ensamble[-1].k = self.k
            self.ensamble[-1].parameter_list = self.parameter_list

        if generate:
            self.generate_channel()

    def apply_channel(self, state):
        state_list = []
        for channel in self.ensamble:
            state_new = channel.apply_channel(state)
            state_list.append(state_new)

        return state_list

    def generate_channel(self):
        for channel in self.ensamble:
            channel.generate_channel()


class ExtractedKrausMap(KrausMap):
    def __init__(
        self,
        d=None,
        rank=None,
        spam=None,
        trainable=True,
        generate=True,
    ):
        KrausMap.__init__(
            self, d=d, rank=rank - 1, spam=spam, trainable=trainable, generate=False
        )
        self.UnitaryMap = UnitaryMap(d=d, trainable=trainable, generate=False)
        _, self.k, _ = generate_ginibre(1, 1, trainable=trainable, complex=False)

        self.parameter_list.extend(self.UnitaryMap.parameter_list)
        self.parameter_list.append(self.k)

        self.kraus = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        KrausMap.generate_channel(self)
        self.UnitaryMap.generate_channel()

        c = 1 / (1 + tf.exp(-self.k))
        c = tf.cast(c, dtype=precision)
        self.kraus = tf.concat(
            [tf.sqrt(c) * self.UnitaryMap.U, tf.sqrt(1 - c) * self.kraus], axis=1
        )

    @property
    def c(self):
        if self.k is None:
            c = None
        else:
            c = 1 / (1 + tf.exp(-self.k))
        return c


class SquaredKrausMap(KrausMap):
    def __init__(
        self,
        d=None,
        rank=None,
        spam=None,
        trainable=True,
        generate=True,
    ):
        KrausMap.__init__(d, rank, spam, trainable, generate=generate)

    def apply_channel(self, state):
        state = KrausMap.apply_channel(state)
        state = KrausMap.apply_channel(state)

        return state

    @property
    def choi(self):
        return channel_to_choi(self)