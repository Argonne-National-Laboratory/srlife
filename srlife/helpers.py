"""
  Helper functions for converting to and from tensor notation
"""

from __future__ import division

import itertools

import numpy as np

mandel = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
mandel_mults = (1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2))


def make_M2T():
    """
    Make the tensor that goes from rank 4 Mandel to rank 4 tensors
    """
    R = np.zeros((3, 3, 3, 3, 6, 6))
    for a in range(6):
        for b in range(6):
            ind_a = itertools.permutations(mandel[a], r=2)
            ind_b = itertools.permutations(mandel[b], r=2)
            ma = mandel_mults[a]
            mb = mandel_mults[b]
            indexes = tuple(ai + bi for ai, bi in itertools.product(ind_a, ind_b))
            for ind in indexes:
                R[ind + (a, b)] = 1.0 / (ma * mb)

    return R


M2T = make_M2T().reshape(81, 36)


def make_usym():
    """
    Make the tensor that goes from Mandel to tensors
    """
    R = np.zeros((3, 3, 6))
    for a in range(6):
        R[mandel[a][0], mandel[a][1], a] = 1.0 / mandel_mults[a]
        R[mandel[a][1], mandel[a][0], a] = 1.0 / mandel_mults[a]

    return R


U = make_usym().reshape(9, 6)


def make_sym():
    """
    Make the tensor that goes from tensors to Mandel
    """
    R = np.zeros((6, 3, 3))
    for a in range(6):
        R[a, mandel[a][0], mandel[a][1]] = mandel_mults[a]

    return R


S = make_sym().reshape(6, 9)


def ms2ts(C):
    """
    Convert a Mandel notation stiffness matrix to a full stiffness tensor.
    """
    Ct = np.zeros((3, 3, 3, 3))
    for a in range(6):
        for b in range(6):
            ind_a = itertools.permutations(mandel[a], r=2)
            ind_b = itertools.permutations(mandel[b], r=2)
            ma = mandel_mults[a]
            mb = mandel_mults[b]
            indexes = tuple(ai + bi for ai, bi in itertools.product(ind_a, ind_b))
            for ind in indexes:
                Ct[ind] = C[a, b] / (ma * mb)

    return Ct


def ms2ts_faster(C):
    """
    Get cute with symmetry
    """
    # pylint: disable=too-many-function-args
    return np.dot(M2T, C.flatten()).reshape(3, 3, 3, 3)


def ts2ms(C):
    """
    Convert a stiffness tensor into a Mandel notation stiffness matrix
    """
    Cv = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            ma = mandel_mults[i]
            mb = mandel_mults[j]
            Cv[i, j] = C[mandel[i] + mandel[j]] * ma * mb

    return Cv


def sym(A):
    """
    Take a symmetric matrix to the Mandel convention vector.
    """
    return np.array(
        [
            A[0, 0],
            A[1, 1],
            A[2, 2],
            np.sqrt(2) * A[1, 2],
            np.sqrt(2) * A[0, 2],
            np.sqrt(2) * A[0, 1],
        ]
    )


def sym_faster(A):
    """
    Take a symmetric tensor to the Mandel convention with multiplication
    """
    return np.dot(S, A.flatten())


def usym(v):
    """
    Take a Mandel symmetric vector to the full matrix.
    """
    return np.array(
        [
            [v[0], v[5] / np.sqrt(2), v[4] / np.sqrt(2)],
            [v[5] / np.sqrt(2), v[1], v[3] / np.sqrt(2)],
            [v[4] / np.sqrt(2), v[3] / np.sqrt(2), v[2]],
        ]
    )


def usym_faster(v):
    """
    Take a tensor to Mandel using multiplication
    """
    return np.dot(U, v).reshape(3, 3)
