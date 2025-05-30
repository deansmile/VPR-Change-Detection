import numpy as np


def segmented_sum(x, s_ind, e_ind):
    """
    Computes the summation along segments x[s_ind:e_ind, ...]

    Parameters
    ----------
    x : a numpy ndarray
    s_ind : a 1d array (n,)
    e_ind : a 1d array (n,)

    Return
    ------
    a numpy ndarray
    """
    x = np.array(x)
    s_ind = np.array(s_ind)
    e_ind = np.array(e_ind)
    max_num = x.shape[0]

    if s_ind.shape != e_ind.shape or len(s_ind.shape) != 1:
        raise ValueError("s_end and e_end must be 1d array.")

    if np.any(s_ind >= max_num) or np.any(e_ind > max_num):
        raise ValueError(
            "\n"
            "s_end must be [ 0, x.shape[axis] )\n"
            "e_end must be [ 0, x.shape[axis] ]"
        )

    if np.any((e_ind - s_ind) < 0):
        raise ValueError("s_ind cannot greater than e_ind")

    # numpy ufunc.reduceat limitation:
    # e_ind cannot equal greater to len(x)
    # the last elements will be added in a few line later
    I = np.nonzero(e_ind == max_num)[0]
    e_ind[I] = max_num - 1

    splits = np.vstack([s_ind, e_ind]).T.flatten()
    r = np.add.reduceat(x, splits, axis=0)[::2]

    # add the last elements back
    J = s_ind[I] != max_num - 1
    r[I[J]] += x.take(-1, axis=0)

    return r


def segmented_count(x, s_ind, e_ind):
    """
    Counts the number of elements along segments x[s_ind:e_ind]

    Parameters
    ----------
    x : a numpy ndarray
    s_ind : a 1d array (n,)
    e_ind : a 1d array (n,)

    Return
    ------
    a numpy ndarray
    """
    x = np.array(x)
    s_ind = np.array(s_ind)
    e_ind = np.array(e_ind)
    max_num = x.shape[0]

    if s_ind.shape != e_ind.shape or len(s_ind.shape) != 1:
        raise ValueError("s_end and e_end must be 1d array.")

    if np.any(s_ind >= max_num) or np.any(e_ind > max_num):
        raise ValueError(
            "\n"
            "s_end must be [ 0, x.shape[axis] )\n"
            "e_end must be [ 0, x.shape[axis] ]"
        )

    if np.any((e_ind - s_ind) < 0):
        raise ValueError("s_ind cannot greater than e_ind")

    I = s_ind == e_ind
    e_ind[I] = e_ind[I] + 1
    return e_ind - s_ind


def segmented_mean(x, s_ind, e_ind):
    """
    Computes the average along segments x[s_ind:e_ind, ...]

    Parameters
    ----------
    x : a numpy ndarray
    s_ind : a 1d array (n,)
    e_ind : a 1d array (n,)

    Return
    ------
    a numpy ndarray
    """
    r = segmented_sum(x, s_ind, e_ind)
    c = segmented_count(x, s_ind, e_ind)

    assert len(r) == len(c)

    shp = [1] * len(r.shape)
    shp[0] = len(c)
    return 1.0 * r / c.reshape(shp)
