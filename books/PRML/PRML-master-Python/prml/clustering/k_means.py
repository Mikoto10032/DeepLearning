import numpy as np
from scipy.spatial.distance import cdist


class KMeans(object):

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X, iter_max=100):
        """
        perform k-means algorithm

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data
        iter_max : int
            maximum number of iterations

        Returns
        -------
        centers : (n_clusters, n_features) ndarray
            center of each cluster
        """
        I = np.eye(self.n_clusters)
        centers = X[np.random.choice(len(X), self.n_clusters, replace=False)]
        for _ in range(iter_max):
            prev_centers = np.copy(centers)
            D = cdist(X, centers)
            cluster_index = np.argmin(D, axis=1)
            cluster_index = I[cluster_index]
            centers = np.sum(X[:, None, :] * cluster_index[:, :, None], axis=0) / np.sum(cluster_index, axis=0)[:, None]
            if np.allclose(prev_centers, centers):
                break
        self.centers = centers

    def predict(self, X):
        """
        calculate closest cluster center index

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data

        Returns
        -------
        index : (sample_size,) ndarray
            indicates which cluster they belong
        """
        D = cdist(X, self.centers)
        return np.argmin(D, axis=1)
