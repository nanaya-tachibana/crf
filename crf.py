# Module for constructing a linear-chain CRF.
# This implementation borrows mostly from Tensorflow CRF module (https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/crf/python/ops/crf.py#L477).
# For an introduction to conditional random field, read https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf.
import numpy as np
import numba
import functools

import mxnet as mx

from mxnet import nd, init
from mxnet.gluon import nn


@numba.jit
def viterbi_decode(transitions, emissions, mask=None):
    """
    Decode the highest scoring sequence of tags outside of mxnet.

    Parameters:
    ----
    transitions: numpy array, shape(num_tags, num_tags)
    emissions: numpy array, shape(seq_length, batch_size, num_tags)
    mask: numpy array, shape(seq_length, batch_size)

    Returns:
    ----
    best_tags_list: list of list
      Each list in best_tags_list is the best path of each sequence.
    """
    seq_legnth, batch_size, num_tags = emissions.shape
    if mask is None:
        mask = np.ones((seq_legnth, batch_size))
    assert mask[0].all()

    sequence_lengths = mask.sum(axis=0).astype(np.int64)
    # list to store the decode paths
    best_tags_list = []
    # (seq_length, batch_size, num_tags)
    viterbi_scores = np.zeros_like(emissions, dtype=np.float32)
    viterbi_scores[0] = emissions[0]
    # (seq_length, batch_size, num_tags)
    viterbi_paths = np.zeros_like(emissions, dtype=np.int64)

    # use dynamic programing to compute the viterbi score
    for i, emission in enumerate(emissions[1:]):
        # (batch_size, num_tags, 1)
        broadcast_log_prob = viterbi_scores[i].reshape((-1, num_tags, 1))
        # (batch_size, num_tags, num_tags)
        score = broadcast_log_prob + transitions
        viterbi_paths[i] = score.argmax(axis=1)
        viterbi_scores[i + 1] = emission + score.max(axis=1)
    # search best path for each batch according to the viterbi score
    # get the best tag for the last emission
    for idx in range(batch_size):
        seq_end_idx = sequence_lengths[idx] - 1
        best_last_tag = viterbi_scores[seq_end_idx, idx].argmax()
        best_tags = [best_last_tag]
        # trace back all best tags based on the last best tag and viterbi path
        for path in np.flip(viterbi_paths[:sequence_lengths[idx] - 1], 0):
            best_last_tag = path[idx][best_tags[-1]]
            best_tags.append(best_last_tag)
        best_tags.reverse()
        best_tags_list.append(best_tags)
    return best_tags_list


def log_sum_exp(F, arr, axis=1):
    """
    Sum arr along axis in log space.

    Parameters:
    ----
    F: mxnet.Symbol or mxnet.ndarray
    arr: any shape
         Input array
    axis: int
          Axis summing along (axis <= len(shape))
    """
    offset = F.max(arr, axis=axis, keepdims=True)
    safe_sum = F.log(F.sum(F.exp(F.broadcast_minus(arr, offset)), axis=axis))
    return F.squeeze(offset, axis=axis) + safe_sum


def _slice(F, t, begin, end, axis=None):
    return F.squeeze(
        F.slice_axis(t, begin=begin, end=end, axis=axis), axis=axis)


def _broadcast_score(F, num_tags, transitions, log_probs):
    """
    Computes the transition scores for every possible path.
    """
    # (batch_size, num_tags, 1)
    broadcast_log_probs = F.reshape(log_probs, shape=(-1, num_tags, 1))
    # (batch_size, num_tags, num_tags)
    return F.broadcast_add(broadcast_log_probs, transitions)


def _crf_unary_score(F, inputs, last_scores):
    """
    Computes the unary scores of tag sequence.
    """
    emission, cur_tag, mask = inputs
    scores = F.pick(emission, cur_tag, axis=1)
    # add emission score for current tag if current tag is not masked
    return [], F.where(mask, scores + last_scores, last_scores)


def _crf_binary_score(F, transitions, inputs, last_scores):
    """
    Computes the binary scores of tag sequence.
    """
    cur_tag, next_tag, mask = inputs
    scores = F.gather_nd(transitions, F.stack(cur_tag, next_tag))
    # add transition score to next tag if next tag is not masked
    return [], F.where(mask, scores + last_scores, last_scores)


def _crf_log_norm(F, num_tags, transitions, inputs, log_probs):
    """
    Computes the normalization of tag sequence.
    """
    emission, mask = inputs
    # (batch_size, num_tags, num_tags)
    transition_scores = _broadcast_score(F, num_tags, transitions, log_probs)
    # (batch_size, num_tags)
    scores = log_sum_exp(F, transition_scores, axis=1)
    return [], F.where(mask, emission + scores, log_probs)


class Crf(nn.HybridBlock):
    """
    Linear-chain conditional random field

    Parameters:
    ----
    num_tags: int
      Number of states.
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
        Computes the log-likelihood of tag sequence.

        Parameters:
        ----
        emissions: shape(seq_length, batch_size, num_tags)
        tags: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)

        Returns:
        ----
        loglikelihood: shape(batch_size, )
        """
        sequence_score = self._log_joint_prob(
            F, emissions, tags, mask, transitions)
        log_norm = self._log_norm(F, emissions, mask, transitions)
        return sequence_score - log_norm

    def _log_joint_prob(self, F, emissions, tags, mask, transitions):
        """
        Computes the unnormalized log prob of tag sequence.

        \sum_{t=0}^{T-1} M(y_{t-1}, y_{t}) + \sum_{t=0}^{T} f(x_{t-1}, y_{t-1})

        Parameters:
        ----
        F: mxnet.Symbol or mxnet.ndarray
        emissions: shape(seq_length, batch_size, num_tags)
        tags: shape(seq_length, batch_size)
        mask: shape(seq_length, batch_size)
        transitions: shape(num_tags, num_tags)

        Returns:
        ----
        log_probs: shape(batch_size, )
        """
        log_probs = F.zeros_like(_slice(F, tags, begin=0, end=1, axis=0))
        # \sum_{t=0}^{T} f(x_{t-1}, y_{t-1})
        _, log_probs = F.contrib.foreach(
            functools.partial(_crf_unary_score, F),
            [emissions, tags, mask], log_probs)
        # \sum_{t=0}^{T-1} M(y_{t-1}, y_{t})
        _, log_probs = F.contrib.foreach(
            functools.partial(_crf_binary_score, F, transitions),
            [F.slice_axis(tags, axis=0, begin=0, end=-1),
             F.slice_axis(tags, axis=0, begin=1, end=None),
             F.slice_axis(mask, axis=0, begin=1, end=None)], log_probs)
        return log_probs

    def _log_norm(self, F, emissions, mask, transitions):
        """
        Computes the normalization of sequence.

        \alpha = \sum_{y_t \in S, y_{t-1} \in S} \
        \sum_{t=0}^{T-1} M(y_{t-1}, y_{t}) + \sum_{t=0}^{T} f(x_{t-1}, y_{t-1})

        Parameters:
        ----
        F: mxnet.Symbol or mxnet.ndarray
        emissions: shape(seq_length, batch_size, num_tags)
        mask: shape(seq_length, batch_size)
        transitions: shape(num_tags, num_tags)

        Returns:
        ----
        log_norm: shape(batch_size, num_tags)
        """
        _, log_probs = F.contrib.foreach(
            functools.partial(_crf_log_norm, F, self.num_tags, transitions),
            [F.slice_axis(emissions, axis=0, begin=1, end=None),
             F.slice_axis(mask, axis=0, begin=1, end=None)],
            _slice(F, emissions, begin=0, end=1, axis=0))
        return log_sum_exp(F, log_probs, axis=1)


def _crf_viterbi_forward(F, num_tags, transitions, inputs, scores):
    emission, mask = inputs
    # (batch_size, num_tags, num_tags)
    transition_scores = _broadcast_score(F, num_tags, transitions, scores)
    # (batch_size, num_tags)
    path = F.argmax(transition_scores, axis=1)

    new_scores = F.where(mask,
                         emission + F.max(transition_scores, axis=1),
                         scores)
    return [path, new_scores], new_scores


def _crf_viterbi_backward(F, inputs, last_tag):
    path, scores, mask = inputs
    best_tag = F.where(mask,
                       F.pick(path, last_tag, axis=1),
                       F.argmax(scores, axis=1))
    return best_tag, best_tag


class ViterbiDecoder(nn.HybridBlock):
    """
    Decode the highest scoring sequence of tags in mxnet.

    Parameters:
    ----
    num_tags: int
      Number of states
    transition: mx.ndarray, shape(num_tags, num_tags)
      Transition matrix of crf.

    Usage:
    decoder = ViterbiDecoder(num_tags, transitions)
    decoder.initialize(init=init.Xavier())
    decoder.hybridize()
    paths, _ = decoder(emissions, mask)
    paths = [path[:m.sum().asscalar()].asnumpy().tolist()
             for path, m in zip(paths.T, mask.T)]
    """
    def __init__(self, num_tags, transitions, prefix='viterbi_'):
        assert num_tags == transitions.shape[0] == transitions.shape[1], \
            f'num_tags: {num_tags}, but shape ' \
            f'of transitions: {transitions.shape}'
        super().__init__(prefix=prefix)
        self.num_tags = num_tags
        with self.name_scope():
            self.transitions = self.params.get_constant(
                'transitions', value=transitions)

    def hybrid_forward(self, F, emissions, mask, transitions):
        (viterbi_paths, viterbi_scores), last_scores = F.contrib.foreach(
            functools.partial(
                _crf_viterbi_forward, F, self.num_tags, transitions),
            [F.slice_axis(emissions, axis=0, begin=1, end=None),
             F.slice_axis(mask, axis=0, begin=1, end=None)],
            _slice(F, emissions, begin=0, end=1, axis=0))
        last_tag = F.argmax(last_scores, axis=1)
        best_path, _ = F.contrib.foreach(
            functools.partial(_crf_viterbi_backward, F),
            [F.SequenceReverse(viterbi_paths),
             F.SequenceReverse(viterbi_scores),
             F.slice(mask, begin=(-1, None), end=(0, None), step=(-1, 1))],
            last_tag)
        best_path = F.concat(last_tag.reshape((1, -1)), best_path, dim=0)
        return F.SequenceReverse(best_path), F.max(last_scores, axis=1)


if __name__ == '__main__':
    seq_length, batch_size, num_tags = 4, 2, 5
    mask = nd.array([[1, 1], [1, 1], [1, 0], [1, 0]])
    tags = nd.array([[0, 1], [2, 4], [3, 1], [1, 0]])
    m = Crf(num_tags)
    m.initialize(init=init.Xavier())
    m.hybridize()
    emissions = nd.array([
        [[0.9383, 0.4889, -0.6731, 0.8728, 1.0554],
         [0.1778, -0.2303, -0.3918, -1.5810, 1.7066]],
        [[-0.4462, 0.7440, 1.5210, 3.4105, -1.1256],
         [-0.3170, -1.0925, -0.0852, -0.0933, 0.6871]],
        [[-0.4462, 0.7440, 1.5210, 3.4105, -1.1256],
         [0.6382, -0.2460, 2.3025, -1.8817, -0.0497]],
        [[-0.8383, 0.0009, -0.7504, 0.1854, 0.6211],
         [0.6382, -0.2460, 2.3025, -1.8817, -0.0497]]])
    m.transitions.set_data(nd.array([
        [-0.0693, -0.1000, 0.0145, 0.0948, 0.0549],
        [-0.0347, 0.0900, -0.0808, -0.0608, -0.0277],
        [-0.0747, -0.0884, -0.0698, 0.0517, -0.0683],
        [0.0845, -0.0411, -0.0849, -0.0357, -0.0408],
        [0.0506, -0.0526, -0.0175, -0.0538, 0.0537]]))
    assert abs(m(emissions, tags, mask).sum().asscalar()
               - (-8.493363)) <= 1e-4
    assert abs(m(emissions, tags, mx.nd.ones_like(tags)).sum().asscalar()
               - (-13.35252)) <= 1e-4

    with mx.autograd.record():
        loss = m(emissions, tags, mask)
    loss.backward()

    transitions = m.transitions.data().asnumpy()
    emissions = emissions.asnumpy()
    mask = mask.asnumpy()

    paths = viterbi_decode(transitions, emissions, mask)
    assert paths == [[0, 3, 3, 4], [4, 4]]

    paths = viterbi_decode(transitions, emissions)
    assert paths == [[0, 3, 3, 4], [4, 4, 2, 2]]
    print('All tests passed')
