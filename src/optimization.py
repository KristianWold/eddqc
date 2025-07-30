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
from loss_functions import *
from quantum_channel import *
from utils import *
from set_precision import *
from spam import povm_fidelity


class Logger:
    def __init__(
        self,
        sample_freq=100,
        loss_function_list=None,
        verbose=True,
    ):
        self.sample_freq = sample_freq
        self.loss_function_list = loss_function_list
        self.verbose = verbose

        self.loss_list = [[] for _ in range(len(loss_function_list))]

    def log(self, other, push=False):
        if other.counter % self.sample_freq == 0 or push:
            other.channel.generate_channel()

            if self.loss_function_list != None:
                inputs_val_list = other.inputs_val
                targets_val_list = other.targets_val

                for inputs_val, targets_val, loss_function, loss_val in zip(
                    other.inputs_val,
                    other.targets_val,
                    self.loss_function_list,
                    self.loss_list,
                ):
                    loss_temp = np.real(
                        loss_function(other.channel, inputs_val, targets_val)
                    )

                    loss_val.append(loss_temp)

            if self.verbose or push:
                print([float(loss[-1]) for loss in self.loss_list])


class ModelQuantumMap:
    def __init__(
        self,
        channel=None,
        loss_function=None,
        optimizer=None,
        logger=None,
    ):
        self.channel = channel
        self.loss_function = loss_function
        self.optimizer = optimizer

        if logger is None:
            logger = Logger(loss_function=loss_function, verbose=False, N=0)
        self.logger = logger

        if not isinstance(self.loss_function, list):
            self.loss_function = [self.loss_function]

    def train(
        self,
        inputs=None,
        targets=None,
        inputs_val=None,
        targets_val=None,
        num_iter=1000,
        N=0,
        verbose=True,
    ):
        self.inputs = inputs
        self.targets = targets
        self.inputs_val = inputs_val
        self.targets_val = targets_val
        self.logger.verbose = verbose
        self.counter = 0

        if N != 0:
            indices = list(range(targets.shape[0]))
        if verbose:
            decorator = tqdm
        else:
            decorator = lambda x: x

        for step in decorator(range(num_iter)):
            if N != 0:
                batch = tf.random.shuffle(indices)[:N]
                inputs_batch = [tf.gather(data, batch, axis=0) for data in inputs]
                targets_batch = tf.gather(targets, batch, axis=0)
            else:
                inputs_batch = inputs
                targets_batch = targets

            self.train_step(inputs_batch, targets_batch)

            self.logger.log(self)
            self.counter += 1

        # print(loss)
        self.channel.generate_channel()
        self.logger.log(self, push=True)

    @tf.function
    def train_step(self, inputs_batch, targets_batch):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.channel.parameter_list)
            self.channel.generate_channel()

            loss = 0
            for loss_function in self.loss_function:
                loss += loss_function(self.channel, inputs_batch, targets_batch)

        grads = tape.gradient(loss, self.channel.parameter_list)
        self.optimizer.apply_gradients(zip(grads, self.channel.parameter_list))

    def zero_optimizer(self):
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))

    def set_loss_function(self, loss_function, zero_optimizer=True):
        self.loss_function = loss_function
        if not isinstance(self.loss_function, list):
            self.loss_function = [self.loss_function]
        if zero_optimizer:
            self.zero_optimizer()

    def spectrum(self, **kwargs):
        return self.channel.spectrum(**kwargs)


class ModelSPAM:
    def __init__(
        self,
        spam=None,
        optimizer=None,
    ):
        self.spam = spam
        self.optimizer = optimizer
        self.d = spam.d

    def train(
        self,
        inputs=None,
        targets=None,
        inputs_val=None,
        targets_val=None,
        spam_target=None,
        num_iter=1000,
        N=0,
        verbose=True,
        sample_freq=100,
    ):
        self.inputs = inputs
        self.targets = targets
        self.inputs_val = inputs_val
        self.targets_val = targets_val
        self.spam_target = spam_target

        self.counter = 0

        self.train_loss_list = []
        self.val_loss_list = []
        self.init_fid_list = []
        self.povm_fid_list = []

        if N != 0:
            indices = list(range(targets.shape[0]))
        if verbose:
            decorator = tqdm
        else:
            decorator = lambda x: x

        for step in decorator(range(num_iter)):
            if N != 0:
                batch = tf.random.shuffle(indices)[:N]
                inputs_batch = tf.gather(inputs, batch, axis=0)
                targets_batch = tf.gather(targets, batch, axis=0)
            else:
                inputs_batch = inputs
                targets_batch = targets

            self.train_step(inputs_batch, targets_batch)

            if self.counter % sample_freq == 0:
                self.spam.generate_SPAM()
                train_loss = np.abs(self.loss(inputs, targets))
                self.train_loss_list.append(train_loss)
                if verbose:
                    print(self.train_loss_list[-1], end=" ")

                if inputs_val is not None:
                    val_loss = np.abs(self.loss(self.inputs_val, self.targets_val))
                    self.val_loss_list.append(val_loss)
                    if verbose:
                        print(self.train_loss_list[-1], end=" ")

                if self.spam_target is not None:
                    init_fid = np.abs(
                        state_fidelity(self.spam.init.init, self.spam_target.init.init)
                    )
                    povm_fid = np.abs(
                        povm_fidelity(self.spam.povm.povm, self.spam_target.povm.povm)
                    )
                    self.init_fid_list.append(init_fid)
                    self.povm_fid_list.append(povm_fid)
                    if verbose:
                        print(self.init_fid_list[-1], self.povm_fid_list[-1], end=" ")

                if verbose:
                    print("")

            self.counter += 1

        self.spam.generate_SPAM()

    def loss(self, inputs_batch, targets_batch):
        N = targets_batch.shape[0]
        outputs_batch = measurement(
            tf.repeat(self.spam.init.init[None, :, :], N, axis=0),
            U_basis=inputs_batch,
            povm=self.spam.povm.povm,
        )
        return self.d * tf.math.reduce_mean((targets_batch - outputs_batch) ** 2)

    @tf.function
    def train_step(self, inputs_batch, targets_batch):
        with tf.GradientTape() as tape:
            self.spam.generate_SPAM()
            loss = self.loss(inputs_batch, targets_batch)

        grads = tape.gradient(loss, self.spam.parameter_list)
        self.optimizer.apply_gradients(zip(grads, self.spam.parameter_list))

    def pretrain(self, num_iter, targets=[None, None], verbose=True):
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        init_target, povm_target = targets
        if init_target is None:
            init_target = init_ideal(self.d)
        if povm_target is None:
            povm_target = povm_ideal(self.d)

        for step in tqdm(range(num_iter), disable=not verbose):
            with tf.GradientTape() as tape:
                self.spam.generate_SPAM()
                loss1 = tf.reduce_mean(tf.abs(self.spam.init.init - init_target) ** 2)
                loss2 = tf.reduce_mean(tf.abs(self.spam.povm.povm - povm_target) ** 2)
                loss = loss1 + loss2

            grads = tape.gradient(loss, self.spam.parameter_list)
            optimizer.apply_gradients(zip(grads, self.spam.parameter_list))
            if verbose:
                print(step, np.abs(loss.numpy()))


def fit_model(
    channel=None,
    spam=None,
    N_map=None,
    N_spam=None,
    loss_function=None,
    num_iter_pretrain=300,
    num_iter_map=2000,
    num_iter_spam=2000,
    filename=None,
    ratio=None,
    verbose=False,
    counts=False,
):
    if not counts:
        inputs_map, targets_map, inputs_spam, targets_spam = pickle.load(
            open(filename, "rb")
        )
    else:
        inputs_map, inputs_spam, counts = pickle.load(open(filename, "rb"))
        targets = counts_to_probs(counts)
        N = inputs_map[0].shape[0]
        targets_map = targets[:N]
        targets_spam = targets[N:]

    if ratio is not None:
        inputs_map, targets_map, _, _ = train_val_split(
            inputs_map, targets_map, ratio=ratio
        )

    if num_iter_pretrain != 0:
        spam.pretrain(num_iter=num_iter_pretrain, verbose=False)

    spam.train(
        inputs=inputs_spam,
        targets=targets_spam,
        num_iter=num_iter_spam,
        N=N_spam,
        verbose=verbose,
    )

    channel.spam = spam
    model = ModelQuantumMap(
        channel=channel,
        loss_function=loss_function,
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        logger=Logger(
            loss_function=ProbabilityMSE(),
            loss_function_val=ProbabilityRValue(),
            verbose=verbose,
        ),
    )

    model.train(
        inputs=inputs_map,
        targets=targets_map,
        inputs_val=inputs_map,
        targets_val=targets_map,
        num_iter=num_iter_map,
        N=N_map,
        verbose=verbose,
    )
    model.optimizer = None
    spam.optimizer = None
    model.inputs = None
    model.targets = None
    model.inputs_val = None
    model.targets_val = None

    return model


def model_remove_misc(model):

    model.optimizer = None

    model.inputs = None
    model.targets = None
    model.inputs_val = None
    model.targets_val = None

    if hasattr(model, "channel"):
        model.channel.spam.optimizer = None

    return model


def model_saver(model, filename, remove_misc=True):
    if not isinstance(model, list):
        model_list = [model]
    else:
        model_list = model

    if remove_misc:
        for model in model_list:
            model_remove_misc(model)

    pickle.dump(model_list, open(filename, "wb"))
