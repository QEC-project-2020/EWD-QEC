import numpy as np
import matplotlib.pyplot as plt
from random import uniform, randint, random
from numba import jit, njit
import random as rand

class Planar_code():
    nbr_eq_classes = 4
    
    def __init__(self, size):
        self.system_size = size
        self.qubit_matrix = np.zeros((2, self.system_size, self.system_size), dtype=np.uint8)
        self.plaquette_defects = np.zeros((size, size-1), dtype=bool)
        self.vertex_defects = np.zeros((size-1, size), dtype=bool)


    def generate_random_error(self, p_error):
        qubits = np.random.uniform(0, 1, size=(2, self.system_size, self.system_size))
        no_error = qubits > p_error
        error = qubits < p_error
        qubits[no_error] = 0
        qubits[error] = 1
        pauli_error = np.random.randint(3, size=(2, self.system_size, self.system_size)) + 1
        self.qubit_matrix[:,:,:] = np.multiply(qubits, pauli_error)
        self.qubit_matrix[1,-1,:] = 0
        self.qubit_matrix[1,:,-1] = 0
        self.syndrom()


    def count_errors(self):
        return _count_errors(self.qubit_matrix)


    def apply_logical(self, operator: int, X_pos=0, Z_pos=0):
        return _apply_logical(self.qubit_matrix, operator, X_pos, Z_pos)


    def apply_stabilizer(self, row: int, col: int, operator: int):
        return _apply_stabilizer(self.qubit_matrix, row, col, operator)


    def apply_random_logical(self):
        return _apply_random_logical(self.qubit_matrix)


    def apply_random_stabilizer(self):
        return _apply_random_stabilizer(self.qubit_matrix)


    def apply_stabilizers_uniform(self, p=0.5):
        return _apply_stabilizers_uniform(self.qubit_matrix, p)


    def define_equivalence_class(self):
        return _define_equivalence_class(self.qubit_matrix)


    def to_class(self, eq: int): # apply_logical_operators i decoders.py
        return _to_class(eq, self.qubit_matrix)


    def syndrom(self):
        # generate vertex excitations (charge)
        # can be generated by z and y errors 
        yz_errors = (self.qubit_matrix == 2) | (self.qubit_matrix == 3) # separate y and z errors from x
        yz_outer = yz_errors[0]
        yz_inner = yz_errors[1]
        # annihilate two defects at the same place in the grid
        charge_vert = yz_outer[1:,:] ^ yz_outer[:-1,:]
        charge_horiz = yz_inner[:-1,:] ^ np.roll(yz_inner[:-1,:], 1, axis=1)
        self.vertex_defects = charge_horiz ^ charge_vert
        
        # generate plaquette excitation (flux)
        # can be generated by x and y errors
        xy_errors = (self.qubit_matrix == 1) | (self.qubit_matrix == 2)
        xy_outer = xy_errors[0]
        xy_inner = xy_errors[1]
        # annihilate two defects at the same place in the grid
        flux_vert = xy_inner[:,:-1] ^ np.roll(xy_inner[:,:-1], 1, axis=0)
        flux_horiz = xy_outer[:,1:] ^ xy_outer[:,:-1]
        self.plaquette_defects = flux_horiz ^ flux_vert


    def plot(self, title, show_eq_class=False):
        # Find all, x y, z errors in both layers
        x_error_outer = np.where(self.qubit_matrix[0,:,:] == 1)
        y_error_outer = np.where(self.qubit_matrix[0,:,:] == 2)
        z_error_outer = np.where(self.qubit_matrix[0,:,:] == 3)

        x_error_inner = np.where(self.qubit_matrix[1,:,:] == 1)
        y_error_inner = np.where(self.qubit_matrix[1,:,:] == 2)
        z_error_inner = np.where(self.qubit_matrix[1,:,:] == 3)

        # Find coordinates of vertex, plaquette defects
        vertex_defect_coordinates = np.where(self.vertex_defects)
        plaquette_defect_coordinates = np.where(self.plaquette_defects)

        xLine = np.linspace(0, self.system_size-1, self.system_size)
        x = range(self.system_size)
        X, Y = np.meshgrid(x,x)
        XLine, YLine = np.meshgrid(x, xLine)

        markersize_qubit = 15
        markersize_excitation = 7
        markersize_symbols = 7
        linewidth = 2

        # Plot grid lines
        ax = plt.subplot(111)
        ax.plot(XLine, -YLine - 0.5, 'black', linewidth=linewidth)
        ax.plot(YLine[:,1:], -XLine[:,1:], 'black', linewidth=linewidth)

        ax.plot(X[:-1,:-1] + 0.5, -Y[1:,1:], 'o', color = 'black', markerfacecolor = 'white', markersize=markersize_qubit+1)
        ax.plot(X, -Y -0.5, 'o', color = 'black', markerfacecolor = 'white', markersize=markersize_qubit+1)

        # all x errors
        ax.plot(x_error_outer[1], -x_error_outer[0] - 0.5, 'o', color = 'r', label="x error", markersize=markersize_qubit)
        ax.plot(x_error_inner[1] + 0.5, -x_error_inner[0] - 1, 'o', color = 'r', markersize=markersize_qubit)
        ax.plot(x_error_outer[1], -x_error_outer[0] - 0.5, 'o', color = 'black', markersize=markersize_symbols, marker=r'$X$')
        ax.plot(x_error_inner[1] + 0.5, -x_error_inner[0] - 1, 'o', color = 'black', markersize=markersize_symbols, marker=r'$X$')

        # all y errors
        ax.plot(y_error_outer[1], -y_error_outer[0] - 0.5, 'o', color = 'blueviolet', label="y error", markersize=markersize_qubit)
        ax.plot(y_error_inner[1] + 0.5, -y_error_inner[0] - 1, 'o', color = 'blueviolet', markersize=markersize_qubit)
        ax.plot(y_error_outer[1], -y_error_outer[0] - 0.5, 'o', color = 'black', markersize=markersize_symbols, marker=r'$Y$')
        ax.plot(y_error_inner[1] + 0.5, -y_error_inner[0] - 1, 'o', color = 'black', markersize=markersize_symbols, marker=r'$Y$')

        # all z errors
        ax.plot(z_error_outer[1], -z_error_outer[0] - 0.5, 'o', color = 'b', label="z error", markersize=markersize_qubit)
        ax.plot(z_error_inner[1] + 0.5, -z_error_inner[0] - 1, 'o', color = 'b', markersize=markersize_qubit)
        ax.plot(z_error_outer[1], -z_error_outer[0] - 0.5, 'o', color = 'black', markersize=markersize_symbols, marker=r'$Z$')
        ax.plot(z_error_inner[1] + 0.5, -z_error_inner[0] - 1, 'o', color = 'black', markersize=markersize_symbols  , marker=r'$Z$')

        # Plot defects
        ax.plot(vertex_defect_coordinates[1], -vertex_defect_coordinates[0] - 1, 'o', color = 'blue', label="charge", markersize=markersize_excitation)
        ax.plot(plaquette_defect_coordinates[1] + 0.5, -plaquette_defect_coordinates[0] - 0.5, 'o', color = 'red', label="flux", markersize=markersize_excitation)
        ax.axis('off')

        # Show equivalence class 
        if show_eq_class:
            ax.set_title('Equivalence class: {}'.format(self.define_equivalence_class()))

        plt.axis('equal')
        plt.savefig('plots/graph_'+str(title)+'.png')
        plt.close()


@njit(cache=True)
def _count_errors(qubit_matrix):
    return np.count_nonzero(qubit_matrix)


# At the moment numba is limited in compiling classes
# So some class functions above are simply wrappers of the compiled functions below
@njit(cache=True)
def _apply_logical(qubit_matrix, operator: int, X_pos=0, Z_pos=0):
    # Have to make copy, else original matrix is changed
    result_qubit_matrix = np.copy(qubit_matrix)

    # Operator is zero means identity, no need to keep going
    if operator == 0:
        return result_qubit_matrix, 0
    size = qubit_matrix.shape[1]
    layer = 0

    error_count = 0

    do_X = (operator == 1 or operator == 2)
    do_Z = (operator == 3 or operator == 2)

    # Helper function
    def qubit_update(row, col, op):
        old_qubit = result_qubit_matrix[layer, row, col]
        new_qubit = old_qubit ^ op
        result_qubit_matrix[layer, row, col] = new_qubit
        if old_qubit and not new_qubit:
            return -1
        elif new_qubit and not old_qubit:
            return 1
        else:
            return 0

    for i in range(size):
        if do_X:
            error_count += qubit_update(X_pos, i, 1)
        if do_Z:
            error_count += qubit_update(i, Z_pos, 3)

    return result_qubit_matrix, error_count


@njit(cache=True)
def _apply_random_logical(qubit_matrix):
    size = qubit_matrix.shape[1]

    # operator to use, 2 (Y) will make both X and Z on the same layer. 0 is identity
    # one operator for each layer
    op = int(random() * 4)

    if op == 1 or op == 2:
        X_pos = int(random() * size)
    else:
        X_pos = 0
    if op == 3 or op == 2:
        Z_pos = int(random() * size)
    else:
        Z_pos = 0

    return _apply_logical(qubit_matrix, op, X_pos, Z_pos)


@njit(cache=True)
def _apply_stabilizer(qubit_matrix, row: int, col: int, operator: int):
    # gives the resulting qubit error matrix from applying (row, col, operator) stabilizer
    # doesn't update input qubit_matrix
    size = qubit_matrix.shape[1]
    if operator == 1:
        # Special cases depending on where the stabilizer lives (square/triangle - in the middle/on the boundary)
        if col == 0:
            qubit_matrix_layers = np.array([0, 0, 1])
            rows = np.array([row, row + 1, row])
            cols = np.array([0, 0, 0])
        elif col == size - 1:
            qubit_matrix_layers = np.array([0, 0, 1])
            rows = np.array([row, row + 1, row])
            cols = np.array([col, col, col - 1])
        else:
            qubit_matrix_layers = np.array([0, 0, 1, 1])
            rows = np.array([row, row + 1, row, row])
            cols = np.array([col, col, col, col - 1])

    elif operator == 3:
        # Special cases depending on where the stabilizer lives (square/triangle - in the middle/on the boundary)
        if row == 0:
            qubit_matrix_layers = np.array([0, 0, 1])
            rows = np.array([0, 0, 0])
            cols = np.array([col, col + 1, col])
        elif row == size - 1:
            qubit_matrix_layers = np.array([0, 0, 1])
            rows = np.array([row, row, row - 1])
            cols = np.array([col, col + 1, col])
        else:
            qubit_matrix_layers = np.array([0, 0, 1, 1])
            rows = np.array([row, row, row, row - 1])
            cols = np.array([col, col + 1, col, col]) 

    # Have to make copy, else original matrix is changed
    result_qubit_matrix = np.copy(qubit_matrix)
    error_count = 0
    for i in range(len(qubit_matrix_layers)):
        old_qubit = qubit_matrix[qubit_matrix_layers[i], rows[i], cols[i]]
        new_qubit = operator ^ old_qubit
        result_qubit_matrix[qubit_matrix_layers[i], rows[i], cols[i]] = new_qubit
        if old_qubit and not new_qubit:
            error_count -= 1
        elif new_qubit and not old_qubit:
            error_count += 1

    return result_qubit_matrix, error_count


@njit(cache=True)
def _apply_random_stabilizer(qubit_matrix):
    size = qubit_matrix.shape[1]
    if rand.random() < 0.5:
        # operator = 1 = x
        return _apply_stabilizer(qubit_matrix,rand.randint(0,size-2),rand.randint(0,size-1),1)
    else: 
        # operator = 3 = z
        return _apply_stabilizer(qubit_matrix,rand.randint(0,size-1),rand.randint(0,size-2),3)


def _apply_stabilizers_uniform(qubit_matrix, p=0.5):
    size = qubit_matrix.shape[1]
    result_qubit_matrix = np.copy(qubit_matrix)
    random_stabilizers = np.random.rand(2, size, size)
    random_stabilizers = np.less(random_stabilizers, p)

    # Remove stabilizers from illegal positions
    #x-operators at bottom
    random_stabilizers[1,size-1,:] = 0
    #z-operators at right edge
    random_stabilizers[0,:,size-1] = 0

    # Numpy magic for iterating through matrix
    it = np.nditer(random_stabilizers, flags=['multi_index'])
    while not it.finished:
        if it[0]:
            op, row, col = it.multi_index
            if op == 0:
                op = 3
            result_qubit_matrix, _ = _apply_stabilizer(result_qubit_matrix, row, col, op)
        it.iternext()
    return result_qubit_matrix


@njit(cache=True)
def _define_equivalence_class(qubit_matrix):
    # of x errors in the first column
    x_errors = np.count_nonzero(qubit_matrix[0,:,0]==1)
    x_errors += np.count_nonzero(qubit_matrix[0,:,0]==2)

    # of z errors in the first row
    z_errors = np.count_nonzero(qubit_matrix[0,0,:]==3)
    z_errors += np.count_nonzero(qubit_matrix[0,0,:]==2)

    # return the parity of the calculated #'s of errors
    return (x_errors % 2) + 2 * (z_errors % 2)


@njit(cache=True)
def _to_class(eq, qubit_matrix):
    # Returns an error chain with same syndrom as qubit_matrix, but in the class eq
    # eq is interpreted as a 2-digit binary number (z x)
    # xor target class with current class, to calculate what operators "connect" them
    diff = eq ^ _define_equivalence_class(qubit_matrix)

    # These lines flip x if z==1
    # This converts a 2-bit eq-class into a 2-bit operator
    mask = 0b10
    xor = (mask & diff) >> 1
    op = diff ^ xor

    # Apply the operator
    qubit_matrix, _ = _apply_logical(qubit_matrix, operator=op, X_pos=0, Z_pos=0)

    return qubit_matrix
