Description
===========
A module provides an implementation of [linear-chain conditional random field](https://en.wikipedia.org/wiki/Conditional_random_field) in mxnet using HybridBlock. This implementation borrows mostly from [Tensorflow CRF module](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/crf/python/ops/crf.py#L477).


Requirements
============

- numpy
- mxnet
- numba


Examples
==========


```Python
    import mxnet as mx
    seq_length, batch_size, num_tags = 4, 2, 5
    mask = nd.array([[1, 1], [1, 1], [1, 0], [1, 0]])
    tags = nd.array([[0, 1], [2, 4], [3, 1], [1, 0]])
    m = Crf(num_tags)
    m.initialize(init=mx.init.Xavier())
    m.hybridize()
    emissions = mx.nd.array([
        [[0.9383, 0.4889, -0.6731, 0.8728, 1.0554],
         [0.1778, -0.2303, -0.3918, -1.5810, 1.7066]],
        [[-0.4462, 0.7440, 1.5210, 3.4105, -1.1256],
         [-0.3170, -1.0925, -0.0852, -0.0933, 0.6871]],
        [[-0.4462, 0.7440, 1.5210, 3.4105, -1.1256],
         [0.6382, -0.2460, 2.3025, -1.8817, -0.0497]],
        [[-0.8383, 0.0009, -0.7504, 0.1854, 0.6211],
         [0.6382, -0.2460, 2.3025, -1.8817, -0.0497]]])
    m.transitions.set_data(mx.nd.array([
        [-0.0693, -0.1000, 0.0145, 0.0948, 0.0549],
        [-0.0347, 0.0900, -0.0808, -0.0608, -0.0277],
        [-0.0747, -0.0884, -0.0698, 0.0517, -0.0683],
        [0.0845, -0.0411, -0.0849, -0.0357, -0.0408],
        [0.0506, -0.0526, -0.0175, -0.0538, 0.0537]]))
```


Computing log likelihood
-----
```Python
    >>> m(emissions, tags, mask)
        [-5.2060437 -3.2873197]
        <NDArray 2 @cpu(0)>
```

Decoding
----
```Python
    >>> transitions = m.transitions.data().asnumpy()
    >>> emissions = emissions.asnumpy()
    >>> mask = mask.asnumpy()
    >>> viterbi_decode(transitions, emissions, mask))
	[[0, 3, 3, 4], [4, 4]]
```

