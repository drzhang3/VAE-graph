import numpy as np


def data_mu_scaler(values_):
    values = np.reshape(values_, [-1, 1])
    # values = np.array(values)
    return (values - np.mean(values)) / np.std(values)


def get_train_data(value, seq_len, step):
    """
    Create x, y train data windows.
    """
    data_x = []
    data_y = []
    for i in range(0, len(value) - seq_len, step):
        x = value[i:i + seq_len + 1]
        y = value[i:i + seq_len + 1]
        data_x.append(x[:-1])
        data_y.append(y[-1])
    return np.array(data_x), np.array(data_y)


def cid_ce(x, normalize):
    """
    This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
    valleys etc.). It calculates the value of

    .. math::

        \\sqrt{ \\sum_{i=0}^{n-2lag} ( x_{i} - x_{i+1})^2 }

    .. rubric:: References

    |  [1] Batista, Gustavo EAPA, et al (2014).
    |  CID: an efficient complexity-invariant distance for time series.
    |  Data Mining and Knowledge Discovery 28.3 (2014): 634-669.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param normalize: should the time series be z-transformed?
    :type normalize: bool

    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if normalize:
        s = np.std(x)
        if s!=0:
            x = (x - np.mean(x))/s
        else:
            return 0.0
