"""
Some basic helpers operating on Lasagne
"""
from collections import OrderedDict
import numpy as np

import theano
import theano.tensor as T

import lasagne as L
import lasagne.layers as LL
import lasagne.nonlinearities as LN
import lasagne.init as LI
import lasagne.updates as LU
import logging


def L2_dist_squared(x, y):
    xsq = T.sqr(x).sum(axis=1).reshape((x.shape[0], 1))
    ysq = T.sqr(y).sum(axis=1).reshape((1, y.shape[0]))
    return xsq + ysq - 2.0 * T.dot(x, y.T) + 1E-06


class GCNNLayer(LL.MergeLayer):
    """
    """

    def __init__(self, incomings, nfilters, nrings=5, nrays=16,
                 W=LI.GlorotNormal(), b=LI.Constant(0.0),
                 normalize_rings=False, normalize_input=False,
                 take_max=True, nonlinearity=L.nonlinearities.rectify, **kwargs):
        super(GCNNLayer, self).__init__(incomings, **kwargs)
        self.nfilters = nfilters
        self.filter_shape = (nfilters, self.input_shapes[0][1], nrings, nrays)
        self.nrings = nrings
        self.nrays = nrays
        self.normalize_rings = normalize_rings
        self.normalize_input = normalize_input
        self.take_max = take_max
        self.nonlinearity = nonlinearity

        self.W = self.add_param(W, self.filter_shape, name="W")

        biases_shape = (nfilters,)
        self.b = self.add_param(b, biases_shape, name="b", regularizable=False)

    def get_output_shape_for(self, input_shapes):
        shp = input_shapes[0]
        nrays = self.nrays
        if self.take_max:
            nrays = 1
        out_shp = (shp[0], self.nfilters * 1 * nrays)

        return out_shp

    def get_output_for(self, inputs, **kwargs):
        y, M = inputs

        if self.normalize_input:
            y /= T.sqrt(T.sum(T.sqr(y), axis=1) + 1e-5).dimshuffle(0, 'x')

        # theano.dot works both for sparse and dense matrices
        desc_net = theano.dot(M, y)

        desc_net = T.reshape(desc_net, (M.shape[1], self.nrings, self.nrays, y.shape[1]))
        desc_net = desc_net.dimshuffle(0, 3, 1, 2)

        if self.normalize_rings:
            # Unit length per ring
            desc_net /= (1e-5 + T.sqrt(T.sum(T.sqr(desc_net), axis=2) + 1e-5).dimshuffle(0, 1, 'x', 2))

        # pad it along the rays axis so that conv2d produces circular
        # convolution along that dimension
        desc_net = T.concatenate([desc_net, desc_net[:, :, :, :-1]], axis=3)

        # output is N x outmaps x 1 x nrays if filter size is the same as
        # input image size prior padding
        y = theano.tensor.nnet.conv.conv2d(desc_net, self.W,
                                           (self.input_shapes[0][0], self.filter_shape[1], self.nrings,
                                            self.nrays * 2 - 1), self.filter_shape)

        if self.take_max:
            # take the max activation along all rotations of the disk
            y = T.max(y, axis=3).dimshuffle(0, 1, 2, 'x')
            # y is now shaped as N x outmaps x 1 x 1

        if self.b is not None:
            y += self.b.dimshuffle('x', 0, 'x', 'x')

        y = y.flatten(2)

        return self.nonlinearity(y)


class COVLayer(LL.Layer):
    def __init__(self, incoming, **kwargs):
        super(COVLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])

    def get_output_for(self, input, **kwargs):
        x = input
        x -= x.mean(axis=0)
        x = T.dot(x.T, x) / (self.input_shape[0] - 1)
        x = x.flatten(2)
        return x


class Unit(LI.Initializer):
    def __init__(self):
        pass

    def sample(self, shape):
        return T.eye(shape[0], shape[1])


class FMAPLayer(LL.MergeLayer):
    def __init__(self, incomings, ref_lbo, neigen, nonlinearity=None, **kwargs):
        super(FMAPLayer, self).__init__(incomings, **kwargs)
        self.phi_n = T.constant(ref_lbo, 'ref')
        self.neigen = neigen
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1])

    def get_output_for(self, inputs, **kwargs):
        f, phi_m = inputs  # f - inputs, phi_m - basis

        # compute A - the input coefficients matrix
        A = desc_coeff(f, phi_m[:, 0: self.neigen])

        # compute B - the reference coefficients matrix
        B = desc_coeff(f, self.phi_n[:, 0: self.neigen])

        # compute C using least-squares: argmin_X( ||X*A - B||^2 )
        self.C = T.transpose(ldiv(T.transpose(A), T.transpose(B)))

        # apply mapping A*C
        Br = T.dot(self.C, A)

        # compute smoothed mapped functions g
        gr = T.dot(self.phi_n[:, 0: self.neigen], Br)

        if self.nonlinearity:
            gr = self.nonlinearity(gr)

        return gr

    def get_fmap(self):
        return self.C


class FMAPLayer_(LL.MergeLayer):
    def __init__(self, incomings, neigen,
                 C=Unit(), nonlinearity=None,
                 **kwargs):
        super(FMAPLayer_, self).__init__(incomings, **kwargs)
        self.neigen = neigen
        self.nonlinearity = nonlinearity
        self.C = self.add_param(C, (neigen, neigen), name="C")

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1])

    def get_output_for(self, inputs, **kwargs):
        x, f = inputs  # f - inputs, phi_m - basis
        # f.shape = Nxl, phi_m.shape = Nxn

        # compute A - the input coefficients matrix
        F = T.transpose(desc_coeff(x, f[:, 0: self.neigen]))
        # apply mapping A*C
        G = T.dot(F, self.C)
        # compute mapped functions g
        g = T.dot(f[:, 0: self.neigen], T.transpose(G))

        if self.nonlinearity:
            g = self.nonlinearity(g)

        return g


def diag_penalty(x):
    return T.sum(T.mean(T.abs_(x), axis=1, keepdims=False) / (T.abs_(T.diag(x)) + .05))


def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))


def categorical_crossentropy_logdomain(log_predictions, targets, nclasses):
    # return -T.sum(theano.tensor.extra_ops.to_one_hot(targets, nclasses) * log_predictions, axis=1)
    # http://deeplearning.net/tutorial/logreg.html#logreg
    return - log_predictions[T.arange(targets.shape[0]), targets]


def desc_coeff(desc, basis):
    return ldiv(basis, desc)


def ldiv(a, b):
    '''
    :return: pinv(a) * b  <=> (a \ b)
    '''
    at = T.transpose(a)
    at_a = T.dot(at, a)
    at_ai = T.nlinalg.matrix_inverse(at_a)
    at_b = T.dot(at, b)
    return T.dot(at_ai, at_b)
