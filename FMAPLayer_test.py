import sys
import os
import numpy as np
import scipy.io
import time

import theano
import theano.tensor as T
import theano.sparse as Tsp

import lasagne as L
import lasagne.layers as LL
import lasagne.objectives as LO
import lasagne.nonlinearities as LN
from lasagne.layers.normalization import batch_norm

sys.path.append('..')
from icnn import utils_lasagne, dataset, snapshotter

base_path = './dataset/FAUST_registrations/data/diam=200/'
nin = 6890
neigen = 30

phi_m_v = dataset.load_matlab_file(os.path.join(base_path, 'lbo', 'tr_reg_080.mat'), 'Phi')
phi_n_v = dataset.load_matlab_file(os.path.join(base_path, 'lbo', 'tr_reg_000.mat'), 'Phi')

phi_m_v = phi_m_v[:, 0: neigen]
phi_n_v = phi_n_v[:, 0: neigen]
f_v = dataset.load_matlab_file('./tr_reg_080.mat', 'desc')
f_v = f_v[:, 0: 1000]
x_ = [f_v, phi_m_v]
phi_n = T.constant(phi_n_v, 'ref')

inp = LL.InputLayer(shape=(None, nin))
lb_op = LL.InputLayer(input_var=T.matrix('lb_op'), shape=(None, None))


# ffn = utils_lasagne.FMAPLayer_([inp, lb_op], neigen=100, nonlinearity=None)
# ffn = utils_lasagne.FMAPLayer([inp, lb_op], ref_lbo=gg, neigen=100, nonlinearity=None)
# output = LL.get_output(ffn)

def ldiv(a, b):
    '''
    :return: pinv(a) * b  <=> (a \ b)
    '''
    at = T.transpose(a)
    at_a = T.dot(at, a)
    at_ai = T.nlinalg.matrix_inverse(at_a)
    at_b = T.dot(at, b)
    return T.dot(at_ai, at_b), at, at_a, at_ai, at_b


f, phi_m = inp.input_var, lb_op.input_var  # f - inputs, phi_m - basis  # f.shape = Nxl, phi_m.shape = Nxn

f = T.printing.Print('f')(f)
phi_m = T.printing.Print('phi_m')(phi_m)

# compute A - the input coefficients matrix
A = utils_lasagne.desc_coeff(f, phi_m[:, 0: neigen])
A = T.printing.Print('A')(A)
# compute B - the reference coefficients matrix
# B, At, AtA, AtAi, AtB = ldiv(phi_n[:, 0: neigen], f)
B = utils_lasagne.desc_coeff(f, phi_n[:, 0: neigen])
B = T.printing.Print('B')(B)
# compute C using least-squares: argmin_X( ||X*A - B||^2 )
C = T.transpose(utils_lasagne.ldiv(T.transpose(A), T.transpose(B)))
C = T.printing.Print('C')(C)
# apply mapping A*C
Br = T.dot(C, A)
Br = T.printing.Print('Br')(Br)
# compute smoothed mapped functions g
output = T.dot(phi_n[:, 0: neigen], Br)

funcs = dict()
funcs['predict'] = theano.function([inp.input_var, lb_op.input_var],
                                   [output, A, B, C, Br],  #, At, AtA, AtAi, AtB],
                                   on_unused_input='warn')

# output_, A_, B_, C_, Br_, gr_, At_, AtA_, AtAi_, AtB_ = funcs['predict'](*x_)
output_, A_, B_, C_, Br_, gr_ = funcs['predict'](*x_)

scipy.io.savemat('tmp.mat',
                 {'A': A_, 'B': B_, 'C': C_, 'Br': Br_, 'gr': output_, 'f': f_v,
                  'phi_m': phi_m_v, 'phi_n': phi_n_v})#, 'At': At_, 'AtA': AtA_, 'AtAi': AtAi_, 'AtB': AtB_})

print('End')
