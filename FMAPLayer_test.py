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

x = dataset.load_matlab_file(os.path.join(base_path, 'lbo', 'tr_reg_000.mat'), 'Phi')
x = [x, x]

nin = 100
inp = LL.InputLayer(shape=(None, nin))
lb_op = LL.InputLayer(input_var=T.matrix('lb_op'), shape=(None, None))
ffn = utils_lasagne.FMAPLayer([inp, lb_op], neigen=100, nonlinearity=None)
output = LL.get_output(ffn)

funcs = dict()
funcs['predict'] = theano.function([inp.input_var, lb_op.input_var], [output], on_unused_input='warn')

tmp = funcs['predict'](*x)

print(np.max(np.abs(tmp[0] - x[0])))