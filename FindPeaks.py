import numpy as np

def indexes(y, thres=0.3, min_dist=1, thres_abs=False, axis=0):
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

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected.
        When using with Pandas DataFrames, iloc should be used to access the values at the returned positions.
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not thres_abs:
        thres = thres * (np.max(y,axis=axis) - np.min(y,axis=axis)) + np.min(y,axis=axis)

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

    # todo : threshold is not correct
    peaks = np.where(
        (np.vstack([dy, np.zeros((dy.shape[1]))]) < 0.0)
        & (np.vstack([np.zeros((dy.shape[1])), dy]) > 0.0)
        & ((y > thres))
    )

    print('s')

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

    return peaks

if __name__ == "__main__":
    import peakutils
    from peakutils.plot import plot as pplot
    # import matplotlib.pyplot as plt
    # centers = (30.5, 75.5)
    # x = np.linspace(0, 300, 301)
    # y = (peakutils.gaussian(x, 4, centers[0], 15) +
    #      peakutils.gaussian(x, 2, centers[1], 15) +
    #      np.random.rand(x.size))
    # indices = indexes(y, thres=0.3)
    # # plt.plot(x, y)
    # pplot(x,y,indices[0])
    # plt.title("Data with noise")
    # plt.show()



    import pandas as pd
    import matplotlib.pyplot as plt
    data = pd.read_csv("c1_commscope_raw.csv", index_col=0)
    data.columns = data.columns.astype(np.float)
    distances = np.array(data.columns)
    temperatures = np.array(data.index)
    temperature_array = np.zeros((len(distances)))
    signal_threshold = 0.8

    x, y = np.meshgrid(distances, temperatures)

    indices = indexes(data.values, thres=signal_threshold, min_dist=1, axis=0)

    # sorted = np.sort(np.array([x[indices], y[indices]]),axis=1)
    sorted = x[indices][np.argsort(x[indices])], y[indices][np.argsort(x[indices])]
    unique = sorted[0][np.unique(sorted[0], return_index=True)[1]], sorted[1][np.unique(sorted[0], return_index=True)[1]]

    fig = plt.figure()
    ax1 = fig.add_subplot(221)


    # plt.scatter(x[indices], y[indices],c='red')
    # plt.scatter(unique[0], unique[1], c='red')
    ax1.plot(temperatures, data.values[:,0])
    # plt.scatter(indexes(data.values[0,:]), thres=signal_threshold, min_dist=1, c='red')
    pplot(temperatures, data.values[:,0], indices[0][indices[1] == 0][-1])

    ax2 = fig.add_subplot(222)
    original_indices = peakutils.indexes(data.values[:,0], thres=signal_threshold, min_dist=1)
    #[26  28  65  92 107 116]
    ax2.plot()
    pplot(temperatures, data.values[:,0], original_indices)

    ax3 = fig.add_subplot(212)
    # x,y = np.meshgrid(distances[:5], temperatures)
    ax3.contourf(x, y, data.values)
    ax3.scatter(x[indices], y[indices],c='red')
    ax3.set_xlim(0,1)



    plt.show()
    print(1)
