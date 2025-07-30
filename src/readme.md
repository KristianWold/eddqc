# Read Me

This the codebase, written entierly in Python with huge help from TensorFlow.

## analysis.py

This file contains functions for various analysis of data. It includes calculation statistics on radial and angular parts of spectra and CSR, as well as analysis for Diluted Unitaries.

## experimental.py

This file contains functions for supporting setup of experiments on quantum computers. This includes setting up Pauli Strings, converting shots to probabilities, calculation marginals of probabilities, and executing circuits on quantum computers.

## kraus_channels.py

This file contains functions for creating parametric Kraus channels. In particular, it contains the function 'isomery_to_kraus', which computes a Kruas map from a unitary operator acting on a system dilated with an ancilla qubits. This effectively traces away the ancilla qubits, resulting in a dissipative channel.

## loss_functions.py

This file contains various loss functions relevant for optmization. This includes loss on probabilities used for process tomography and loss on spectra for Diluted Unitaries.

## optimization.py

This contains functions for optimizing parametric models on experimental data.

## quantum_channel.py

This contains various helper functions for computing things related to channels, such as Choi matrices, super operators, distance between channels, concatenation of channels, etc.

## quantum_circuit.py

Various quantum circuits.

## set_precision.py

Used for setting global precision for all calculations.

## spam.py

Functions for computing SPAM errors.

## spectrum.py

Functions for computing things related to quantum spectra, such as spectra, CSR, mean spacing between eigenvalues, etc.

## utils.py

Various utility functions.