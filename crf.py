import functools

import numpy as np
import mxnet as mx

from mxnet import nd, init
from mxnet.gluon import nn


def log_sum_exp(F, arr, axis=1):
    """
    Sum arr along axis in log space.

    Parameters:
    ----
    F: mx.ndarray or mx.symbol
    arr: any shape
         Input array
    axis: int
          Axis summing along (axis <= len(shape))
    """
    offset = F.max(arr, axis=axis, keepdims=True)
    safe_sum = F.log(F.sum(F.exp(F.broadcast_minus(arr, offset)), axis=axis))
    return F.squeeze(offset) + safe_sum


def _slice(F, t, begin, end):
    return F.squeeze(F.slice(t, begin=begin, end=end))


def _broadcast_score(F, num_tags, transitions, log_probs):
    # (batch_size, num_tags, 1)
    broadcast_log_probs = F.reshape(log_probs, shape=(-1, num_tags, 1))
    # (batch_size, num_tags, num_tags)
    return F.broadcast_add(broadcast_log_probs, transitions)


def _crf_viterbi_forward(F, num_tags, transitions, inputs, scores):
    emission, mask = inputs
    # (batch_size, num_tags, num_tags)
    transition_scores = _broadcast_score(F, num_tags, transitions, scores)
    # (batch_size, num_tags)
    path = F.argmax(transition_scores, axis=1)

    new_scores = F.where(mask,
                         emission + F.max(transition_scores, axis=1),
                         scores)
    return path, new_scores


def _crf_viterbi_backward(F, path, last_tag):
    best_tag = F.pick(path, last_tag, axis=1)
    return best_tag, best_tag


class ViterbiDecoder(nn.HybridBlock):

    def __init__(self, num_tags, transitions, prefix='viterbi_'):
        super().__init__(prefix=prefix)
        self.num_tags = num_tags
        with self.name_scope():
            self.transitions = self.params.get_constant(
                'transitions', value=transitions)

    def hybrid_forward(self, F, emissions, mask, transitions):
        viterbi_paths, last_scores = F.contrib.foreach(
            functools.partial(_crf_viterbi_forward, F, num_tags, transitions),
            [F.slice(emissions, begin=(1, None), end=(None, None)),
             F.slice(mask, begin=(1, None), end=(None, None))],
            _slice(F, emissions, begin=(0, None), end=(1, None)))
        last_tag = F.argmax(last_scores, axis=1)
        best_path, _ = F.contrib.foreach(
            functools.partial(_crf_viterbi_backward, F),
            F.SequenceReverse(viterbi_paths), last_tag)
        best_path = F.concat(last_tag.reshape((1, -1)), best_path, dim=0)
        return F.SequenceReverse(best_path) * mask, F.max(last_scores, axis=1)


def _crf_unary_score(F, inputs, states):
    emission, cur_tag, mask = inputs
    # add emission score for current tag if current tag is not masked
    scores = F.pick(emission, cur_tag, axis=1) * mask
    return [], states + scores


def _crf_binary_score(F, transitions, inputs, states):
    cur_tag, next_tag, mask = inputs
    # add transition score to next tag if next tag is not masked
    scores = F.gather_nd(transitions, F.stack(cur_tag, next_tag)) * mask
    return [], states + scores


def _crf_log_norm(F, num_tags, transitions, inputs, log_probs):
    emission, mask = inputs
    # (batch_size, num_tags, num_tags)
    transition_scores = _broadcast_score(F, num_tags, transitions, log_probs)
    # (batch_size, num_tags)
    scores = log_sum_exp(F, transition_scores, axis=1)
    return [], F.where(mask, emission + scores, log_probs)


class Crf(nn.HybridBlock):
    """
    Conditional random field
    """
    def __init__(self, num_tags, prefix='crf_'):
        super().__init__(prefix=prefix)
        self.num_tags = num_tags
        with self.name_scope():
            self.transitions = self.params.get('transitions',
                                               shape=(num_tags, num_tags),
                                               init=mx.init.Uniform(0.1))

    def hybrid_forward(self, F, emissions, tags, mask, transitions):
        """
        emissions: shape(seq_length, batch_size, num_tags)
        tags: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        sequence_score = self._joint_loglikelihood(
            F, emissions, tags, mask, transitions)
        log_norm = self._loglikelihood(F, emissions, mask, transitions)
        return sequence_score - log_norm

    def _joint_loglikelihood(self, F, emissions, tags, mask, transitions):
        """
        emissions: shape(seq_length, batch_size, num_tags)
        tags: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        """
        llh = F.zeros_like(_slice(
            F, tags, begin=(0, None), end=(1, None)))

        _, llh = F.contrib.foreach(
            functools.partial(_crf_unary_score, F),
            [emissions, tags, mask], llh)

        cur_tags = F.slice_axis(tags, axis=0, begin=0, end=-1)
        next_tags = F.slice_axis(tags, axis=0, begin=1, end=None)
        _, llh = F.contrib.foreach(
            functools.partial(_crf_binary_score, F, transitions),
            [cur_tags, next_tags,
             F.slice_axis(mask, axis=0, begin=1, end=None)], llh)
        return llh

    def _loglikelihood(self, F, emissions, mask, transitions):
        """
        emissions: shape(seq_length, batch_size, num_tags)
        mask: shape(seq_length, batch_size)
        """
        _, log_probs = F.contrib.foreach(
            functools.partial(_crf_log_norm, F, self.num_tags, transitions),
            [F.slice(emissions, begin=(1, None), end=(None, None)),
             F.slice(mask, begin=(1, None), end=(None, None))],
            _slice(F, emissions, begin=(0, None), end=(1, None)))
        return log_sum_exp(F, log_probs, axis=1)


if __name__ == '__main__':
    seq_length, batch_size, num_tags = 3, 2, 5
    mask = nd.array([[1, 1], [1, 1], [1, 0]])
    tags = nd.array([[0, 1], [2, 4], [3, 1]])
    m = Crf(num_tags)
    m.initialize(init=init.Xavier())
    m.hybridize()
    emissions = nd.array([
        [[0.9383, 0.4889, -0.6731, 0.8728, 1.0554],
         [0.1778, -0.2303, -0.3918, -1.5810, 1.7066]],
        [[-0.4462, 0.7440, 1.5210, 3.4105, -1.1256],
         [-0.3170, -1.0925, -0.0852, -0.0933, 0.6871]],
        [[-0.8383, 0.0009, -0.7504, 0.1854, 0.6211],
         [0.6382, -0.2460, 2.3025, -1.8817, -0.0497]]])
    m.transitions.set_data(mx.nd.array([
        [-0.0693, -0.1000, 0.0145, 0.0948, 0.0549],
        [-0.0347, 0.0900, -0.0808, -0.0608, -0.0277],
        [-0.0747, -0.0884, -0.0698, 0.0517, -0.0683],
        [0.0845, -0.0411, -0.0849, -0.0357, -0.0408],
        [0.0506, -0.0526, -0.0175, -0.0538, 0.0537]]))

    assert abs(m(emissions, tags, mask).sum().asscalar()
               - (-8.072865)) <= 1e-4
    assert abs(m(emissions, tags, mx.nd.ones_like(tags)).sum().asscalar()
               - (-10.966491)) <= 1e-4

    # with mx.autograd.record():
    #     loss = m(emissions, tags, mask)
    # loss.backward()

    transitions = m.transitions.data()
    decoder = ViterbiDecoder(num_tags, transitions)
    decoder.initialize(init=init.Xavier())
    decoder.hybridize()

    paths, scores = decoder(emissions, mask)
    assert paths.asnumpy().T.astype(np.int).tolist() == \
        [[0, 3, 4], [4, 4, 0]]

    scores = scores.asnumpy()
    mask = mx.nd.ones_like(tags)
    paths, scores = decoder(emissions, mask)
    assert paths.asnumpy().T.astype(np.int).tolist() == \
        [[0, 3, 4], [4, 4, 2]]
    scores = scores.asnumpy()
    assert abs(scores[0] - 5.0239) <= 1e-4
    assert abs(scores[1] - 4.7324) <= 1e-4
    print('All tests passed')
