from itertools import combinations
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def uniform_vector_nbi(N, M):
    H1 = 1
    while len(list(combinations([i+1 for i in range(H1+M-1)], M-1))) <= N:
        H1 = H1 + 1

    n_vector = len(list(combinations([i+1 for i in range(H1+M-1)], M-1)))
    W = np.array(list(combinations([i+1 for i in range(H1+M-1)], M-1))) - \
        np.repeat(np.array([[i for i in range(M-2+1)]]), n_vector, axis=0) -1

    W = (np.hstack((W, np.zeros((W.shape[0], 1)) + H1)) - np.hstack((np.zeros((W.shape[0], 1)), W))) / H1
    # a = np.hstack(W, np.zeros((W.shape[0], 1))) + H1

    if H1 < M:
        H2 = 0
        while len(list(combinations([i+1 for i in range(H1+M-1)], M-1))) + \
                len(list(combinations([i+1 for i in range(H2+M)], M-1))) <= N:

            H2 = H2 + 1

        if H2 > 0:
            n_vector_W2 = len(list(combinations([i+1 for i in range(H2+M-1)], M-1)))
            W2 = np.array(list(combinations([i+1 for i in range(H2+M-1)], M-1))) - \
                 np.repeat(np.array([[i for i in range(M-2+1)]]), n_vector_W2, axis=0) - 1
            W2 = (np.hstack((W2, np.zeros((W2.shape[0], 1)) + H2)) - np.hstack((np.zeros((W2.shape[0], 1)), W2))) / H2
            W = np.vstack((W, W2/2 + 1/(2*M)))

    W = np.maximum(W, 1.0e-06)

    print(W)
    return W


def get_neighbour_vector(w_lambda, T):
    w_euclidean_distance = euclidean_distances(w_lambda, w_lambda)
    sort_ind = np.argsort(w_euclidean_distance, axis=0)
    # print("sort: ", sort_ind)
    neighbours_vec = sort_ind[:, 0:T]
    return neighbours_vec

def display_w(W):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = W[:, 0]
    y = W[:, 1]
    z = W[:, 2]

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.scatter(x, y, z, marker='.', s=50, label='', color='r')
    plt.show()


def main():
    # needn't consider the relationship of population(N) and number of lambda vector
    # the vector will divide for every individual
    w = uniform_vector_nbi(70, 3)
    print(w.shape)
    neighour_vec = get_neighbour_vector(w, 4)
    print("neig", neighour_vec)
    display_w(w)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()


