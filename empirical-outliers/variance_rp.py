"""
Implements anomaly detection based on conformance scores with application to streamed data
using path signatures.

Reference:
Cochrane, T., Foster, P., Lyons, T., & Arribas, I. P. (2020).
Anomaly detection on streamed data. arXiv preprint arXiv:2006.03487.
"""

import itertools
from joblib import Memory, Parallel, delayed
import sys

import iisignature
import numpy as np
from tqdm import tqdm

from shuffle import shuffle
from variance import _sig, _get_basis, _build_row


def _build_matrix(basis, E, enable_progress=True, **parallel_kwargs):
    A = np.zeros((len(basis), len(basis)))

    pbar = tqdm(basis, total=len(basis), disable=not enable_progress,
                desc="Building shuffle matrix")
    A = np.array(Parallel(**parallel_kwargs)(delayed(_build_row)(w, basis, E) for w in pbar))

    try:
        A_eigvals, A_eigvecs = np.linalg.eig(A)
        if (min(np.abs(A_eigvals)) < 1e-10):
            raise NotImplementedError("Cannot handle rank deficient covariance matrix yet")

    except (NotImplementedError, np.linalg.LinAlgError) as error:
        print(error)
        print("An exception occured when computing eigenvalues")
        sys.exit(1)

    return A_eigvals, A_eigvecs

def _prepare(corpus, order, enable_progress=True, **parallel_kwargs):
    dim = corpus[0].shape[1]
    basis = _get_basis(dim, order)
    basis_extended = _get_basis(dim, 2 * order)

    sigs = np.array(Parallel(**parallel_kwargs)(delayed(_sig)(p, 2 * order)
                                                for p in tqdm(corpus,
                                                              disable=not enable_progress,
                                                              desc="Computing signatures")))
    E = dict(zip(basis_extended, np.mean(sigs, axis=0)))

    A_eigvals, A_eigvecs = _build_matrix(basis, E, enable_progress, **parallel_kwargs)

    return sigs, A_eigvals, A_eigvecs


def variance_rp(paths, corpus, order, projected_dim, cache_dir=None,
             enable_progress=True, **parallel_kwargs):
    r"""
    Compute conformance scores for streams in a testing collection, given a collection of
    streams in a training corpus and based on using signatures of a specified order as the
    feature map. Caches results on disk.

    Parameters
    ----------
    paths: iterable
        Collection of streams for which to compute comformance scores. Each element in the
        collection is an N_i x M array, where N_i is the number of observations in the ith
        stream and where M is dimensionality of each observation.
    corpus: iterable
        Collection of streams forming the training corpus. Each element in the collection
        is an N_i x M array, where N_i is the number of observations in the ith stream
        and where M is dimensionality of each observation.
    order: int
        Desired signature order
    projected_dim: int
        Desired projected dimension
    cache_dir: str
        Directory for caching results of the function call
        (defaults to None, which disables caching)
    enable_progress: bool
        Whether to enable tqdm progress bars (defaults to True)
    \**parallel_kwargs:
        Additional keyword arguments (e.g. n_jobs) are passed to joblib.Parallel, thus
        influencing parallel execution. For additional information, please refer to
        https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
        Parallel execution is disabled by default.
    """
    sigs, A_eigvals, A_eigvecs= Memory(cache_dir, verbose=0).cache(_prepare)(corpus, order,
                                                               enable_progress,
                                                               **parallel_kwargs)

    sig_tmp = _sig(paths[0], order) # Just to use in the next line to get the length
    sigs_order = sigs[:, :len(sig_tmp)]

    # R = S * \Sigma^{-1/2} * U^T, where A = U \Sigma U^T, and S is a scaled Gaussian matrix
    random_projection_mat = np.dot(
        np.dot(
            np.random.randn(projected_dim, np.shape(sigs_order)[1]),  np.diag(
                np.sqrt(np.reciprocal(A_eigvals)))
                )
        , A_eigvecs.T)
    projected_sigs = np.dot(sigs_order, random_projection_mat.T)   

    res = []
    for path in tqdm(paths, disable=not enable_progress, desc="Computing variances"):
        sig = _sig(path, order)
        a = np.dot(sig, random_projection_mat.T) - projected_sigs
        res.append(np.diag(a.dot(a.T)).min()) # This minimal distance neighbour can be done more efficiently
    return res
