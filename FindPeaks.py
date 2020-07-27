import numpy as np


def rolling_mean(arr, n=1, axis=0):
    if axis == 0:
        arr = arr.T
    rows = len(arr)
    stack = np.hstack((np.zeros((rows, 1)), arr))
    cumsum = np.cumsum(stack, axis=1)
    result = (cumsum[:, n:] - cumsum[:, :-n]) / float(n)
    if axis == 0:
        return result.T
    else:  # axis == 1
        return result


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

    peaks = np.where(
        (np.vstack([dy, np.zeros((dy.shape[1]))]) < 0.0)
        & (np.vstack([np.zeros((dy.shape[1])), dy]) > 0.0)
        & ((y > thres))
    )

    # peaks[0] gives row location
    # peaks[1] gives column location

    # handle multiple peaks, respecting the minimum distance
    if peaks[0].size > 1 and min_dist > 1:
        # peaks[0].size > 1 --> more than one peak
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
        # twod_peaks = [[peaks[0][peaks[1] == i]] for i in range(y.shape[1])]
        # highest_to_lowest_peak_indices = [np.argsort(y[:,i][peaks[0][peaks[1] == i]])[::-1] for i in range(y.shape[1])]
        highest_to_lowest = [peaks[0][peaks[1] == i][np.argsort(y[:, i][peaks[0][peaks[1] == i]])[::-1]] for i in
                             range(y.shape[1])]

        # method 1
        args_sorted_from_highest_to_lowest = []

        # # method 2
        # highest = [[], []]
        for i in range(y.shape[1]):
            sub_peaks = peaks[0][peaks[1] == i]
            # method 1
            args_sorted_from_highest_to_lowest.append(sub_peaks[np.argsort(y[:, i][sub_peaks])][::-1])

            # # method 2
            # highest[1].extend([i]*len(sub_peaks))
            # highest[0].extend(sub_peaks[np.argsort(y[:, i][sub_peaks])[::-1]])
            # # generates a list similar to np.where where [0] is row location, and [1] is column location

        rem = np.ones(y.shape, dtype=bool)
        rem[peaks] = False
        #

        # for peak in highest:
        #     if not rem[peak]:
        #         sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
        #         rem[sl] = True
        #         rem[peak] = False

        for i, sub_highest_to_lowest in enumerate(args_sorted_from_highest_to_lowest):
            for peak in sub_highest_to_lowest:
                if not rem[:, i][peak]:
                    sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                    rem[sl,i] = True
                    rem[peak,i] = False



        # todo 1d works
        p = np.arange(y.shape[0])[~rem[:, 0]]

        # todo : convert to 2d
        peaks = np.mgrid[0:y.shape[0],0:y.shape[1]].reshape(2,-1)[~rem]
        # peaks = np.arange(y.shape)[~rem]

    if last_values_only:
        along_x = 1
        if axis == 1:
            along_x = 0
        return [peaks[0][peaks[1] == i][-1] for i in range(y.shape[along_x])]
    else:
        return peaks


if __name__ == "__main__":
    from peakutils.plot import plot as pplot
    import peakutils
    import pandas as pd
    import matplotlib.pyplot as plt

    data = pd.read_csv("sample_data.csv", index_col=0)
    data.columns = data.columns.astype(np.float)
    x_values = np.array(data.columns)
    y_values = np.array(data.index)
    signal_threshold = 0.8

    # 3d data background flattening
    values = data.values - np.min(data.values, axis=1)[:, None]

    # Smooth data along row-axis
    n = 25
    values = rolling_mean(values, n, 0)
    y_values = rolling_mean(y_values[None, :], n, 1)[0]

    # Smooth data along column axis
    n = 1
    values = rolling_mean(values, n, 1)
    x_values = rolling_mean(x_values[None, :], n, 1)[0]

    # minimum distance for each peak
    min_dist = 6

    # meshgrid x and y coordinates
    x, y = np.meshgrid(x_values, y_values)

    # Find peak indices (vectorized)
    indices = indexes(values, thres=signal_threshold, min_dist=min_dist, axis=0)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.set_title('>Threshold = 0.8')
    ax1.plot(y_values, values[:, 0])
    pplot(y_values, values[:, 0], indices[0][indices[1] == 0])

    ax2 = fig.add_subplot(222)
    ax2.set_title('Last Values only')
    indices = indexes(values, thres=signal_threshold, min_dist=min_dist, axis=0, last_values_only=True)
    ax2.plot()
    pplot(y_values, values[:, 0], [indices[0]])

    ax3 = fig.add_subplot(212)
    ax3.set_title('Peak search for 3d data')
    ax3.contourf(x, y, values)
    t = y_values[indexes(values, thres=signal_threshold, min_dist=min_dist, axis=0, last_values_only=True)]
    ax3.plot(x_values, t, c='#ff000080')

    plt.show()
