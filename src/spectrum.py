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
from utils import *
from set_precision import *
from copy import copy


class Spectrum:

    def __init__(
        self,
        spectrum_tensor,
        is_complex=False,
        keep_real=True,
        keep_unity=True,
        tol=1e-4,
    ):

        if isinstance(spectrum_tensor, list):
            spectrum_tensor_list = [
                spectrum_.get_spectrum() for spectrum_ in spectrum_tensor
            ]
            spectrum_tensor = tf.concat(spectrum_tensor_list, axis=0)
            self.set_spectrum(spectrum_tensor)

        self.set_spectrum(spectrum_tensor)
        self.is_complex = is_complex
        self.keep_real = keep_real
        self.keep_unity = keep_unity
        self.tol = tol

    def __call__(self, is_complex=None, keep_real=None, keep_unity=None, tol=None):

        # Create a shallow copy of the current object, with the option to modify the attributes
        self_copy = copy(self)

        if is_complex is not None:
            self_copy.is_complex = is_complex
        if keep_real is not None:
            self_copy.keep_real = keep_real
        if keep_unity is not None:
            self_copy.keep_unity = keep_unity
        if tol is not None:
            self_copy.tol = tol

        return self_copy

    def set_spectrum(self, spectrum_tensor):
        if len(spectrum_tensor.shape) == 1:
            spectrum_tensor = tf.expand_dims(spectrum_tensor, axis=1)

        if len(spectrum_tensor.shape) == 2:
            if spectrum_tensor.shape[1] == 1:
                spectrum_tensor = tf.concat(
                    [tf.math.real(spectrum_tensor), tf.math.imag(spectrum_tensor)],
                    axis=1,
                )
                spectrum_tensor = tf.cast(spectrum_tensor, dtype=precision)

        self.spectrum_tensor = spectrum_tensor
        self.num_eigenvalues = spectrum_tensor.shape[0]

    def get_spectrum(self, is_complex=None, keep_real=None, keep_unity=None, tol=None):
        if is_complex is None:
            is_complex = self.is_complex
        if keep_real is None:
            keep_real = self.keep_real
        if keep_unity is None:
            keep_unity = self.keep_unity
        if tol is None:
            tol = self.tol

        spectrum_mod = self.spectrum_tensor

        if not keep_unity:
            mask = tf.abs(spectrum_mod[:, 0] - 1.0) > self.tol
            spectrum_mod = spectrum_mod[mask]

        if not keep_real:
            mask = tf.abs(spectrum_mod[:, 1]) > self.tol
            spectrum_mod = spectrum_mod[mask]

        if is_complex:
            spectrum_mod = spectrum_mod[:, 0] + 1j * spectrum_mod[:, 1]

        return spectrum_mod

    def csr(self):
        return complex_spacing_ratio(self)

    def plot_circle(self, figsize=(10, 10)):
        t = np.linspace(0, 2 * np.pi, 1000)
        circle = np.array([np.cos(t), np.sin(t)]).T
        plt.figure(figsize=figsize)
        plt.plot(circle[:, 0], circle[:, 1], color="black")

    def plot(self, marker="o", markersize=10):
        spectrum_tensor = tf.math.real(self.get_spectrum(is_complex=False))
        plt.plot(
            spectrum_tensor[:, 0], spectrum_tensor[:, 1], marker, markersize=markersize
        )


def channel_spectrum(
    channel, is_complex=False, keep_real=True, keep_unity=True, tol=1e-4
):
    so = channel.superoperator

    eig, _ = tf.linalg.eig(so)
    eig = tf.expand_dims(eig, axis=1)

    spectrum = Spectrum(
        eig, is_complex=is_complex, keep_real=keep_real, keep_unity=keep_unity, tol=tol
    )

    return spectrum


def channel_spectrum_numpy(
    channel, use_coords=True, keep_real=True, keep_unity=True, tol=1e-4
):
    so = channel.superoperator

    eig = np.linalg.eigvals(so)
    eig = np.expand_dims(eig, axis=1)

    # remove eigenvalues on real line
    if not keep_real:
        mask = np.abs(np.imag(eig)) > tol
        eig = np.expand_dims(eig[mask], axis=1)

    # remove eigenvalues close to unity
    if not keep_unity:
        mask = np.abs(np.real(eig) - 1) > tol
        eig = np.expand_dims(eig[mask], axis=1)

    if use_coords:
        x = np.real(eig)
        y = np.imag(eig)
        eig = np.concatenate([x, y], axis=1)

    return eig


def choi_spectrum(channel):
    eig, _ = tf.linalg.eig(channel.choi)
    eig = tf.expand_dims(eig, axis=1)

    return eig


def normalize_spectrum(spectrum, scale=1):
    spectrum = spectrum.numpy()
    idx = np.argmax(np.linalg.norm(spectrum, axis=1))
    spectrum[idx] = (0, 0)

    max = np.max(np.linalg.norm(spectrum, axis=1))
    spectrum = scale / max * spectrum

    spectrum[idx] = (1, 0)
    spectrum = tf.cast(tf.convert_to_tensor(spectrum), dtype=precision)

    return spectrum


def complex_spacing_ratio(spectrum, verbose=False, log=True):
    if isinstance(spectrum, Spectrum):
        spectrum = np.array(spectrum.get_spectrum(is_complex=True, keep_real=False))
    d = len(spectrum)

    z_list = []
    if verbose:
        decorator = tqdm
    else:
        decorator = lambda x: x

    for i in decorator(range(d)):
        idx_NN = (i + 1) % d
        dist_NN = float("inf")

        idx_NNN = (i + 2) % d
        dist_NNN = float("inf")

        for j in range(d):
            if j != i:
                dist = np.abs(spectrum[i] - spectrum[j])
                if dist < dist_NN:
                    dist_NNN = dist_NN
                    idx_NNN = idx_NN

                    dist_NN = dist
                    idx_NN = j

                elif dist < dist_NNN:
                    dist_NNN = dist
                    idx_NNN = j

            z = (spectrum[idx_NN] - spectrum[i]) / (spectrum[idx_NNN] - spectrum[i])
        z_list.append(z)

    z_list = np.array(z_list)
    # stack real and imaginary part of z_list along second axis
    z_list = np.stack([np.real(z_list), np.imag(z_list)], axis=1)
    z_list = Spectrum(z_list, is_complex=True, keep_real=False, keep_unity=False)

    return z_list


def spacing_ratio(spectrum):
    d = len(spectrum)
    z_list = []
    for i in tqdm(range(d)):
        idx_NN = i
        dist_NN = float("inf")

        idx_NNN = i
        dist_NNN = float("inf")

        for j in range(d):
            if j != i:
                dist = np.angle(spectrum[i] - spectrum[j])
                if dist < dist_NN:
                    dist_NNN = dist_NN
                    idx_NNN = idx_NN

                    dist_NN = dist
                    idx_NN = j

                if (dist > dist_NN) and (dist < dist_NNN):
                    dist_NNN = dist
                    idx_NNN = j

        z = np.angle(spectrum[i] - spectrum[idx_NN]) / np.angle(
            spectrum[i] - spectrum[idx_NNN]
        )
        z_list.append(z)


def distance_spacing_ratio(spectrum, verbose=False):
    d = len(spectrum)
    spectrum = np.array(spectrum)[:, 0]
    z_list = []
    if verbose:
        decorator = tqdm
    else:
        decorator = lambda x: x
    log_spectrum = np.log(spectrum)
    s = mean_spacing(log_spectrum)
    rho = unfolding(log_spectrum, 4.5 * s)

    for i in decorator(range(d)):
        dist_NN = float("inf")
        dist_NNN = float("inf")

        for j in range(d):
            if j != i:
                dist = np.abs(log_spectrum[i] - log_spectrum[j])
                if dist < dist_NN:
                    dist_NNN = dist_NN

                    dist_NN = dist

                if (dist > dist_NN) and (dist < dist_NNN):
                    dist_NNN = dist

        z = dist_NNN / dist_NN
        z = z * np.sqrt(rho[i])
        z_list.append(z)

    return np.array(z_list)


def unfolding(spectrum, sigma):
    N = spectrum.shape[0]
    spectrum = np.array(spectrum)
    diff = np.abs(spectrum.reshape(-1, 1) - spectrum.reshape(1, -1))
    expo = -1 / (2 * sigma**2) * diff**2
    rho = 1 / (2 * np.pi * sigma**2 * N) * np.sum(np.exp(expo), axis=1)
    return rho


def mean_spacing(spectrum):

    if isinstance(spectrum, list):
        spectrum_list = spectrum
        mean_spacing_list = [mean_spacing(spectrum) for spectrum in spectrum_list]
        return np.mean(mean_spacing_list)

    if isinstance(spectrum, Spectrum):
        spectrum = np.array(spectrum.get_spectrum(is_complex=True, keep_real=True))

    d = len(spectrum)
    ms_list = []
    for i in range(d):
        dist_NN = float("inf")

        for j in range(d):
            if j != i:
                dist = np.abs(spectrum[i] - spectrum[j])
                if dist < dist_NN:
                    dist_NN = dist

        ms_list.append(dist_NN)

    return np.mean(ms_list)


def coat_spectrum(spectrum, sigma=0.1, grid_size=100):
    """Coat each eigenvalue with a Gaussian distribution."""
    if isinstance(spectrum, Spectrum):
        spectrum = np.array(spectrum.get_spectrum(is_complex=True))

    x_grid = np.linspace(-1, 1, grid_size)
    y_grid = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    rho = 0
    for eig in spectrum:
        rho += np.exp(-((X - eig.real) ** 2 + (Y - eig.imag) ** 2) / (2 * sigma**2))

    return rho


def hopkins_statistic(spectrum, split=10):
    angles = np.angle(spectrum)
    N = len(angles)
    m = N // split

    idx = np.random.choice(N, m, replace=False)
    X = angles[idx]
    Y = np.random.uniform(0, 2 * np.pi, m)
    distance_X = np.abs(X.reshape(-1, 1) - angles.reshape(1, -1))
    distance_X[distance_X == 0] = np.inf
    u = np.min(distance_X, axis=1)

    distance_Y = np.abs(Y.reshape(-1, 1) - angles.reshape(1, -1))
    w = np.min(distance_Y, axis=1)

    u_sum = np.sum(u)
    w_sum = np.sum(w)

    hs = u_sum / (u_sum + w_sum)

    return hs, u_sum, w_sum


def pauli_string_decomp(eigenvector):
    d = len(eigenvector)
    I = tf.cast([[1, 0], [0, 1]], dtype=precision)
    X = tf.cast([[0, 1], [1, 0]], dtype=precision)
    Y = tf.cast([[0, -1j], [1j, 0]], dtype=precision)
    Z = tf.cast([[1, 0], [0, -1]], dtype=precision)

    pauli_single = []
    pauli_double = []
    for i in range(d):
        pass


def spectrum_distance(spectrum_a, spectrum_b, sigma=0.1):
    """Distance measure between spectra"""

    def overlap(spectrum_a, spectrum_b):
        aa = tf.math.reduce_sum(spectrum_a * spectrum_a, axis=1, keepdims=True)
        bb = tf.math.reduce_sum(spectrum_b * spectrum_b, axis=1, keepdims=True)
        ab = tf.matmul(spectrum_a, spectrum_b, adjoint_b=True)

        expo = aa - 2 * ab + tf.transpose(bb)
        sum = 1 / np.sqrt(sigma) * tf.math.reduce_mean(tf.math.exp(-expo / sigma**2))

        return sum

    spectrum_a = spectrum_a.get_spectrum(is_complex=False, keep_unity=False)
    spectrum_b = spectrum_b.get_spectrum(is_complex=False, keep_unity=False)

    dist = overlap(spectrum_a, spectrum_a)
    dist += -2 * overlap(spectrum_a, spectrum_b)
    dist += overlap(spectrum_b, spectrum_b)

    return dist


def generate_integrable_spectrum(d):
    # sample d-2 complex numbers uniformly in the unit disk
    d2 = d**2
    spectrum = [1, np.random.uniform(-1, 1)]

    for i in range((d2 - 2) // 2):
        theta = np.random.uniform(0, np.pi)
        r = np.sqrt(np.random.uniform(0, 1))
        eig = r * np.exp(1j * theta)
        spectrum.extend([eig, np.conj(eig)])

    spectrum = np.array(spectrum, dtype=np.complex128)
    spectrum = np.vstack([spectrum.real, spectrum.imag]).T
    spectrum = tf.convert_to_tensor(spectrum, dtype=tf.complex128)
    spectrum = Spectrum(spectrum)

    return spectrum


def spectrum_to_radial(spectrum, concat=False):
    if isinstance(spectrum, list):
        spectrum_list = spectrum
        radial_list = [spectrum_to_radial(spectrum) for spectrum in spectrum_list]
        if concat:
            radial_list = np.concatenate(radial_list)
        return radial_list

    if isinstance(spectrum, Spectrum):
        spectrum = spectrum.get_spectrum(is_complex=True, keep_unity=False)
    radial = np.abs(spectrum)
    return radial


def spectrum_to_angular(spectrum, concat=False):
    if isinstance(spectrum, list):
        spectrum_list = spectrum
        angular_list = [spectrum_to_angular(spectrum) for spectrum in spectrum_list]
        if concat:
            angular_list = np.concatenate(angular_list)
        return angular_list

    if isinstance(spectrum, Spectrum):
        spectrum = spectrum.get_spectrum(is_complex=True, keep_unity=False)
    angular = np.angle(spectrum)
    return angular
