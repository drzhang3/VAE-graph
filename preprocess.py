import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def data_mu_scaler(values_):
    values = np.reshape(values_, [-1, 1])
    # values = np.array(values)
    return (values - np.mean(values)) / np.std(values)


def binarization(matrix):
    shape = np.shape(matrix)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if matrix[i][j] < -0.02:
                matrix[i][j] = 0
            else:
                matrix[i][j] = 1
    return matrix


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


def drawDAG(matrix):
    G = nx.DiGraph()
    #nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    #G.add_nodes_from(nodes)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i,j]:
                G.add_edge(i+1, j+1)
    #G=G.to_directed()
    pos = nx.spring_layout(G)
    nx.draw(G,pos, with_labels=True, edge_color='b', node_color='g', node_size=1000)
    plt.show()


def sigmoid(matrix):
    shape = np.shape(matrix)
    mat = np.zeros([shape[0], shape[0]])
    for i in range(shape[0]):
        for j in range(shape[1]):
            mat[i][j] = 1/(1+np.exp(-matrix[i][j]))
    return mat


def mask_data(data):
    # mask = np.random.choice([0, 1], size=np.shape(data[0]), p=[.1, .9])
    mask = np.ones(np.shape(data)[0])
    prop = int(0.05*np.shape(data)[0])
    mask[:prop] = 0
    np.random.shuffle(mask)
    return mask*np.array(data)
