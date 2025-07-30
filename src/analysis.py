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

from utils import *
from set_precision import *
from quantum_tools import *
from experimental import *


def angular_dist(spectrum_list):
    angular_list = [spectrum_to_angular(spectrum) for spectrum in spectrum_list]
    angular = np.concatenate(angular_list)
    return angular


def radial_dist(spectrum_list):
    radial_list = [spectrum_to_radial(spectrum) for spectrum in spectrum_list]
    radial = np.concatenate(radial_list)
    return radial


def hist_ensamble(data_list, bins=15, density=True, error=False):
    hist_density_list = []

    for data in data_list:
        hist_density, bins_ = np.histogram(data, bins=bins, density=density)
        hist_density_list.append(hist_density)

    center_bins = (bins_[:-1] + bins_[1:]) / 2

    hist_density_mean = np.mean(hist_density_list, axis=0)
    ensemble_std = np.std(hist_density_list, axis=0, ddof=1)

    if error == True:
        length = len(data_list)
        ensemble_std = ensemble_std / np.sqrt(length)

    return center_bins, hist_density_mean, ensemble_std


def scatterplot_stats(radial, angular):

    if isinstance(radial, list):
        radial = np.concatenate(radial)
        angular = np.concatenate(angular)

    r_mean = np.mean(radial)
    a_mean = -np.mean(np.cos(angular))

    return r_mean, a_mean


def scatterplot_bootstrap(csr_bootstrap):
    r_list = []
    a_list = []
    for csr in csr_bootstrap:
        r = np.mean(spectrum_to_radial(csr))
        a = np.mean(-np.cos(spectrum_to_angular(csr)))

        r_list.append(r)
        a_list.append(a)
    
    r_mean = np.mean(r_list, axis=0)
    r_std = np.std(r_list, axis=0, ddof=1)

    a_mean = np.mean(a_list, axis=0)
    a_std = np.std(a_list, axis=0, ddof=1)

    return r_mean, r_std, a_mean, a_std


def find_outer_inner_R(spectrum_list, tail_num=10):
    if isinstance(spectrum_list, list):
        radial_list = np.array(
            [np.sort(spectrum_to_radial(spectrum)) for spectrum in spectrum_list]
        )
        R_minus = np.mean(radial_list[:, 0])
        R_minus_std = np.std(radial_list[:, 0])
        R_plus = np.mean(radial_list[:, -1])
        R_plus_std = np.std(radial_list[:, -1])
    else:
        radial_list = np.sort(spectrum_to_radial(spectrum_list))
        R_minus = np.mean(radial_list[:tail_num])
        R_minus_std = np.std(radial_list[:tail_num])
        R_plus = np.mean(radial_list[-tail_num:])
        R_plus_std = np.std(radial_list[-tail_num:])

    return R_plus, R_minus, R_plus_std, R_minus_std


def annulus_distance(spectrum1, spectrum2):
    angular1 = spectrum_to_angular(spectrum1)
    a_mean1 = np.mean(angular1)
    a_std1 = np.std(angular1)

    angular2 = spectrum_to_angular(spectrum2)
    a_mean2 = np.mean(angular2)
    a_std2 = np.std(angular2)

    radial1 = spectrum_to_radial(spectrum1)
    r_mean1 = np.mean(radial1)
    r_std1 = np.std(radial1)

    radial2 = spectrum_to_radial(spectrum2)
    r_mean2 = np.mean(radial2)
    r_std2 = np.std(radial2)

    distance = (
        np.abs(r_mean1 - r_mean2)
        + np.abs(r_std1 - r_std2)
        + np.abs(a_mean1 - a_mean2)
        + np.abs(a_std1 - a_std2)
    )

    return distance


def model_to_csr(model_list):
    spectrum_list = []
    csr_integrable_list = []
    for model in model_list:
        channel = model.channel
        spectrum = channel_spectrum(channel)

        csr = complex_spacing_ratio(spectrum)

        spectrum_list.append(spectrum)
        csr_integrable_list.append(csr)

    return csr_integrable_list




# ---------------------------------#
# Sector Symmetries
# ---------------------------------#


def Z_i(i, n):
    I = np.eye(2, dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    return kron(*[I if j != i else Z for j in range(n)])


def Q_n(n):
    return sum([Z_i(i, n) for i in range(n)])


def Q_diff(n):
    I_d = np.eye(2**n, dtype=np.complex128)
    Q = Q_n(n)
    Q_d = 0.5 * (kron(I_d, Q) - kron(Q, I_d))
    return Q_d
