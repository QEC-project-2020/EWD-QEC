import numpy as np
from random import random
from numba import njit
import random as rand

# XYZ^2 code for ZZ vertical link stabilizers

class xyz_code():
    nbr_eq_classes = 4

    def __init__(self, size):
        self.system_size = size
        self.qubit_matrix = np.zeros((2*self.system_size, self.system_size), dtype=np.uint8)

    def generate_random_error(self, px, py, pz):
        p_x = px
        p_y = py
        p_z = pz
        size = self.system_size
        for i in range(2*size):
            for j in range(size):
                q = 0
                r = rand.random()
                if r <= p_z:
                    q = 3
                elif p_z < r <= (p_z + p_x):
                    q = 1
                elif (p_z + p_x) < r <= (p_z + p_x + p_y):
                    q = 2
                self.qubit_matrix[i, j] = q

    def chain_lengths(self):
        nx = np.count_nonzero(self.qubit_matrix[:, :] == 1)
        ny = np.count_nonzero(self.qubit_matrix[:, :] == 2)
        nz = np.count_nonzero(self.qubit_matrix[:, :] == 3)
        return nx, ny, nz

    def count_errors(self):
        return _count_errors(self.qubit_matrix)

    def apply_logical(self, operator: int, X_pos=0, Z_pos=0):
        return _apply_logical(self.qubit_matrix, operator, X_pos, Z_pos)

    def apply_stabilizer(self, row: int, col: int, operator: int):
        return _apply_stabilizer(self.qubit_matrix, row, col, operator)

    def apply_random_stabilizer(self):
        return _apply_random_stabilizer(self.qubit_matrix)

    def define_equivalence_class(self):
        return _define_equivalence_class(self.qubit_matrix)

    def apply_stabilizers_uniform(self, p=0.5):
        return _apply_stabilizers_uniform(self.qubit_matrix, p)

    def to_class(self, eq):
        eq_class = self.define_equivalence_class()
        op = eq_class ^ eq
        return self.apply_logical(op)[0]


@njit('(uint8[:,:],)')
def _count_errors(qubit_matrix):
    return np.count_nonzero(qubit_matrix)


@njit('(uint8[:,:], int64, int64, int64)')
def _apply_logical(qubit_matrix, operator: int, X_pos=0, Z_pos=0):
    result_qubit_matrix = np.copy(qubit_matrix)

    n_eq = [0, 0, 0, 0]

    if operator == 0:
        return result_qubit_matrix, (0, 0, 0)

    size = qubit_matrix.shape[1]
    op = 0

#     # most likely logical operators for Z-biased noise
    
    do_X = (operator == 1 or operator == 2)
    do_Z = (operator == 3 or operator == 2)

    if do_X:
        for i in range(2*size):
            old_qubit = result_qubit_matrix[i, X_pos]
            if X_pos % 2 == 0:
                if i % 4 == 0:
                    op = 1
                elif i % 4 == 1:
                    op = 2
                elif i % 4 == 2:
                    op = 3
                elif i % 4 == 3:
                    op = 0
            else:
                if i % 4 == 0:
                    op = 0
                elif i % 4 == 1:
                    op = 3
                elif i % 4 == 2:
                    op = 1
                elif i % 4 == 3:
                    op = 2
            new_qubit = op ^ old_qubit
            result_qubit_matrix[i, X_pos] = new_qubit
            n_eq[old_qubit] -= 1
            n_eq[new_qubit] += 1
    if do_Z:
        for i in range(size):
            old_qubit = result_qubit_matrix[2*i, i]
            op = 3
            new_qubit = op ^ old_qubit
            result_qubit_matrix[2*i, i] = new_qubit
            n_eq[old_qubit] -= 1
            n_eq[new_qubit] += 1
    
#     # most likely logical operators for X-biased noise

#     do_X = (operator == 1)
#     do_Z = (operator == 3)
#     do_Y = (operator == 2)

#     if do_X:
#         for i in range(2*size-2):
#             old_qubit = result_qubit_matrix[i, 0]
#             op = 1
#             new_qubit = op ^ old_qubit
#             result_qubit_matrix[i, 0] = new_qubit
#             n_eq[old_qubit] -= 1
#             n_eq[new_qubit] += 1
#         old_qubit = result_qubit_matrix[2*size-1, 1]
#         op = 3
#         new_qubit = op ^ old_qubit
#         result_qubit_matrix[2*size-1, 1] = new_qubit
#         n_eq[old_qubit] -= 1
#         n_eq[new_qubit] += 1
#     if do_Z:
#         for i in range(2):
#             for j in range(size-1):
#                 old_qubit = result_qubit_matrix[2*size-1-i, j]
#                 op = 1
#                 new_qubit = op ^ old_qubit
#                 result_qubit_matrix[2*size-1-i, j] = new_qubit
#                 n_eq[old_qubit] -= 1
#                 n_eq[new_qubit] += 1
#         old_qubit = result_qubit_matrix[2*size-1, size-1]
#         op = 3
#         new_qubit = op ^ old_qubit
#         result_qubit_matrix[2*size-1, size-1] = new_qubit
#         n_eq[old_qubit] -= 1
#         n_eq[new_qubit] += 1
#     if do_Y:
#         for i in range(2*size):
#             for j in range(size):
#                 old_qubit = result_qubit_matrix[i, j]
#                 op = 1
#                 new_qubit = op ^ old_qubit
#                 result_qubit_matrix[i, j] = new_qubit
#                 n_eq[old_qubit] -= 1
#                 n_eq[new_qubit] += 1

    return result_qubit_matrix, (n_eq[1], n_eq[2], n_eq[3])


@njit('(uint8[:,:], int64, int64, int64)')
def _apply_stabilizer(qubit_matrix, row: int, col: int, operator: int):

    size = qubit_matrix.shape[1]
    result_qubit_matrix = np.copy(qubit_matrix)
    n_eq = [0, 0, 0, 0]

    if operator == 1:
        qarray = [[2*row, col], [2*row + 1, col + 1], [2*row + 2, col + 1], [2*row + 3, col + 1], [2*row + 2, col], [2*row + 1, col]]
        opr = [2, 3, 1, 2, 3, 1]
        j = 0
        for i in qarray:
            old_qubit = result_qubit_matrix[i[0], i[1]]
            new_qubit = opr[j] ^ old_qubit
            j += 1
            result_qubit_matrix[i[0], i[1]] = new_qubit
            n_eq[old_qubit] -= 1
            n_eq[new_qubit] += 1
    if operator == 2:
        if col == 0:
            qarray = [[0, 2*row + 1], [1, 2*row + 2], [0, 2*row + 2]]
            opr = [3, 2, 1]
            j = 0
            for i in qarray:
                old_qubit = result_qubit_matrix[i[0], i[1]]
                new_qubit = opr[j] ^ old_qubit
                j += 1
                result_qubit_matrix[i[0], i[1]] = new_qubit
                n_eq[old_qubit] -= 1
                n_eq[new_qubit] += 1
        elif col == 1:
            qarray = [[4*row + 2, size - 1], [4*row + 3, size - 1], [4*row + 4, size - 1]]
            opr = [2, 1, 3]
            j = 0
            for i in qarray:
                old_qubit = result_qubit_matrix[i[0], i[1]]
                new_qubit = opr[j] ^ old_qubit
                j += 1
                result_qubit_matrix[i[0], i[1]] = new_qubit
                n_eq[old_qubit] -= 1
                n_eq[new_qubit] += 1
        elif col == 2:
            qarray = [[2*size - 1, 2*row], [2*size - 2, 2*row], [2*size - 1, 2*row + 1]]
            opr = [1, 2, 3]
            j = 0
            for i in qarray:
                old_qubit = result_qubit_matrix[i[0], i[1]]
                new_qubit = opr[j] ^ old_qubit
                j += 1
                result_qubit_matrix[i[0], i[1]] = new_qubit
                n_eq[old_qubit] -= 1
                n_eq[new_qubit] += 1
        elif col == 3:
            qarray = [[4*row + 1, 0], [4*row + 2, 0], [4*row + 3, 0]]
            opr = [3, 1, 2]
            j = 0
            for i in qarray:
                old_qubit = result_qubit_matrix[i[0], i[1]]
                new_qubit = opr[j] ^ old_qubit
                j += 1
                result_qubit_matrix[i[0], i[1]] = new_qubit
                n_eq[old_qubit] -= 1
                n_eq[new_qubit] += 1
    if operator == 3:
        qarray = [[2*row, col], [1 + 2*row, col]]
        opr = [3, 3]
        j = 0
        for i in qarray:
            old_qubit = result_qubit_matrix[i[0], i[1]]
            new_qubit = opr[j] ^ old_qubit
            j += 1
            result_qubit_matrix[i[0], i[1]] = new_qubit
            n_eq[old_qubit] -= 1
            n_eq[new_qubit] += 1

    return result_qubit_matrix, (n_eq[1], n_eq[2], n_eq[3])


@njit('(uint8[:,:],)')
def _apply_random_stabilizer(qubit_matrix):
    size = qubit_matrix.shape[1]
    rows1 = int((size-1)*random())
    cols1 = int((size-1)*random())
    rows2 = int(((size - 1)/2) * random())
    cols2 = int(4 * random())
    rows3 = int(size * random())
    cols3 = int(size * random())
    phalf = (size**2 - (size-1)**2 - 1)/(size**2-1)
    if rand.random() > 0.5:
        if rand.random() > phalf:
            # operator = 1 = full (xyzxyz) stabilizer
            return _apply_stabilizer(qubit_matrix, rows1, cols1, 1)
        else:
            # operator = 2 = half (xyz) stabilizer
            return _apply_stabilizer(qubit_matrix, rows2, cols2, 2)
    else:
        # operator = 3 = (zz) stabilizer
        return _apply_stabilizer(qubit_matrix, rows3, cols3, 3)


def _apply_stabilizers_uniform(qubit_matrix, p=0.5):
    size = qubit_matrix.shape[1]
    result_qubit_matrix = np.copy(qubit_matrix)

    # Apply full stabilizers
    random_stabilizers = np.random.rand(size-1, size-1)
    random_stabilizers = np.less(random_stabilizers, p)
    it = np.nditer(random_stabilizers, flags=['multi_index'])
    while not it.finished:
        if it[0]:
            row, col = it.multi_index
            result_qubit_matrix, _ = _apply_stabilizer(result_qubit_matrix, row, col, 1)
        it.iternext()

    # Apply half stabilizers
    random_stabilizers = np.random.rand(int((size - 1)/2), 4)
    random_stabilizers = np.less(random_stabilizers, p)
    it = np.nditer(random_stabilizers, flags=['multi_index'])
    while not it.finished:
        if it[0]:
            row, col = it.multi_index
            result_qubit_matrix, _ = _apply_stabilizer(result_qubit_matrix, row, col, 2)
        it.iternext()

    # Apply link stabilizers
    random_stabilizers = np.random.rand(size, size)
    random_stabilizers = np.less(random_stabilizers, p)
    it = np.nditer(random_stabilizers, flags=['multi_index'])
    while not it.finished:
        if it[0]:
            row, col = it.multi_index
            result_qubit_matrix, _ = _apply_stabilizer(result_qubit_matrix, row, col, 3)
        it.iternext()

    return result_qubit_matrix


@njit('(uint8[:,:],)')
def _define_equivalence_class(qubit_matrix):
    size = qubit_matrix.shape[1]
    x_errors = 0
    z_errors = 0

    for i in range(2 * size):
        if i % 4 == 0:
            if qubit_matrix[i, 0] == 2 or qubit_matrix[i, 0] == 3:
                z_errors += 1
        elif i % 4 == 1:
            if qubit_matrix[i, 0] == 1 or qubit_matrix[i, 0] == 3:
                z_errors += 1
        elif i % 4 == 2:
            if qubit_matrix[i, 0] == 1 or qubit_matrix[i, 0] == 2:
                z_errors += 1

    for i in range(2 * size):
        if i % 4 == 1:
            if qubit_matrix[0, int((i-1)/2)] == 1 or qubit_matrix[0, int((i-1)/2)] == 2:
                x_errors += 1
        elif i % 4 == 2:
            if qubit_matrix[1, int(i/2)] == 2 or qubit_matrix[1, int(i/2)] == 3:
                x_errors += 1
        elif i % 4 == 3:
            if qubit_matrix[0, int((i-1)/2)] == 1 or qubit_matrix[0, int((i-1)/2)] == 3:
                x_errors += 1

    # return the parity of the calculated #'s of errors
    if x_errors % 2 == 0:
        if z_errors % 2 == 0:
            return 0
        else:
            return 3
    else:
        if z_errors % 2 == 0:
            return 1
        else:
            return 2
