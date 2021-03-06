import numpy as np


# vectorized version of Peakutils.Indexes

def indexes(y, thres=0.3, min_dist=1, thres_abs=False, axis=0, last_values_only=False):
    """Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.
    thres_abs: boolean
        If True, the thres value will be interpreted as an absolute value, instead of
        a normalized threshold.
    axis : int [0, 1]
        Gets the peaks along rows (0) or columns (1).
    last_values_only : boolean
        If True, returns only one value per row (axis = 0) or column (axis = 1).

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected.
        When using with Pandas DataFrames, iloc should be used to access the values at the returned positions.
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not thres_abs:
        thres = thres * (np.max(y, axis=axis) - np.min(y, axis=axis)) + np.min(y, axis=axis)

    min_dist = int(min_dist)

    # check if data is 1D, else make it 2D
    if len(y.shape) == 1:
        y = y.reshape(y.shape[0], 1)

    # compute first order difference
    dy = np.diff(y, axis=axis)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros = np.where(dy == 0)
    # zeros[0] gives row location
    # zeros[1] gives column location

    # check if the signal is totally flat
    if len(zeros[0]) == len(y) - 1:
        # just return the first value as the maximum
        return np.array([0])


    # if len(zeros[0]):
    #     # compute first order difference of zero indexes
    #     zeros_diff = np.diff(zeros)


    #     # check when zeros are not chained together
    #     zeros_diff_not_one, = np.add(np.where(zeros_diff != 1), 1)
    #     # make an array of the chained zero indexes
    #     zero_plateaus = np.split(zeros, zeros_diff_not_one)
    #
    #     # fix if leftmost value in dy is zero
    #     if zero_plateaus[0][0] == 0:
    #         dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
    #         zero_plateaus.pop(0)
    #
    #     # fix if rightmost value of dy is zero
    #     if len(zero_plateaus) and zero_plateaus[-1][-1] == len(dy) - 1:
    #         dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
    #         zero_plateaus.pop(-1)
    #
    #     # for each chain of zero indexes
    #     for plateau in zero_plateaus:
    #         median = np.median(plateau)
    #         # set leftmost values to leftmost non zero values
    #         dy[plateau[plateau < median]] = dy[plateau[0] - 1]
    #         # set rightmost and middle values to rightmost non zero values
    #         dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    # find the peaks by using the first order difference

    peaks = np.where(
        (np.vstack([dy, np.zeros((dy.shape[1]))]) < 0.0)
        & (np.vstack([np.zeros((dy.shape[1])), dy]) > 0.0)
        & ((y > thres))
    )

    # which row in first column are peaks... (Below)
    # peaks[0][peaks[1] == 0] ==> gives [26  28  65  92 107 116]
    # first data should have peaks ==> [26  28  65  92 107 116]

    # peaks[0] gives row location
    # peaks[1] gives column location

    # handle multiple peaks, respecting the minimum distance
    # if peaks.size > 1 and min_dist > 1:
    #     # y[peaks] gives intensity
    #     # argsort y[peaks] gives the indices to sort y[peaks]
    #     #   *gives row index
    #     # peaks[1]
    #     #   *gives column index
    #     # peaks[argsort(y[peaks])] sorts peak position
    #
    #     # peaks[0][np.argsort(y[peaks])]
    #     # [ 6 95 93 57 55  9 87 64 79 81 77 52 68 83 61 71 13 74 50 48 16 43 21 36, 38 33 25 27 30]
    #
    #     highest_by_arguments = np.argsort(y[peaks], axis=axis)[::-1]
    #     highest = peaks[np.argsort(y[peaks])][::-1]
    #     rem = np.ones(y.shape, dtype=bool)
    #     rem[peaks] = False
    #
    #     for peak in highest:
    #         if not rem[peak]:
    #             sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
    #             rem[sl] = True
    #             rem[peak] = False
    #
    #     peaks = np.arange(y.size)[~rem]

    if last_values_only:
        along_x = 1
        if axis == 1:
            along_x = 0
        return [peaks[0][peaks[1] == i][-1] for i in range(y.shape[along_x])]
    else:
        return peaks


if __name__ == "__main__":
    signal_threshold = 0.5
    x_axis = np.linspace(0, 1000, 500)
    y_axis = np.linspace(0, 100, 200)
    values = np.random.random((200, 500))
    indices = indexes(values, thres=signal_threshold, min_dist=1, axis=0)

    max_values = y_axis[np.array([indices[0][indices[1] == i][-1] for i, dist in enumerate(x_axis)])]
