from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
from sklearn.preprocessing import KernelCenterer
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.utils.validation import check_array
import numpy as np

class Isomap(BaseEstimator, TransformerMixin):
    """Isomap Embedding

    Non-linear dimensionality reduction through Isometric Mapping

    Read more in the :ref:`User Guide <isomap>`.

    Adapted from scikit-learn implementation by Matthew Farrell

    Parameters
    ----------
    n_neighbors : integer
        number of neighbors to consider for each point.

    n_components : integer
        number of coordinates for the manifold

    eigen_solver : ['auto'|'arpack'|'dense']
        'auto' : Attempt to choose the most efficient solver
        for the given problem.

        'arpack' : Use Arnoldi decomposition to find the eigenvalues
        and eigenvectors.

        'dense' : Use a direct solver (i.e. LAPACK)
        for the eigenvalue decomposition.

    tol : float
        Convergence tolerance passed to arpack or lobpcg.
        not used if eigen_solver == 'dense'.

    max_iter : integer
        Maximum number of iterations for the arpack solver.
        not used if eigen_solver == 'dense'.

    path_method : string ['auto'|'FW'|'D']
        Method to use in finding shortest path.

        'auto' : attempt to choose the best algorithm automatically.

        'FW' : Floyd-Warshall algorithm.

        'D' : Dijkstra's algorithm.

    neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
        Algorithm to use for nearest neighbors search,
        passed to neighbors.NearestNeighbors instance.

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.

    kernel_pca_ : object
        `KernelPCA` object used to implement the embedding.

    training_data_ : array-like, shape (n_samples, n_features)
        Stores the training data.

    nbrs_ : sklearn.neighbors.NearestNeighbors instance
        Stores nearest neighbors instance, including BallTree or KDtree
        if applicable.

    dist_matrix_ : array-like, shape (n_samples, n_samples)
        Stores the geodesic distance matrix of training data.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import Isomap
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = Isomap(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)

    References
    ----------

    .. [1] Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. A global geometric
           framework for nonlinear dimensionality reduction. Science 290 (5500)
    """

    def __init__(self, n_neighbors=5, radius=None, n_components=2, eigen_solver='auto',
                 tol=0, max_iter=None, path_method='auto',
                 neighbors_algorithm='auto', n_jobs=None):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs
        self.radius = radius
        self.distance_graph = None
        self.connected_comps = None

    def _fit_transform(self, X, decompose_graph):
        X = check_array(X, accept_sparse='csr')
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm,
                                      n_jobs=self.n_jobs)
        self.nbrs_.fit(X)
        self.training_data_ = self.nbrs_._fit_X
        self.kernel_pca_ = KernelPCA(n_components=self.n_components,
                                     kernel="precomputed",
                                     eigen_solver=self.eigen_solver,
                                     tol=self.tol, max_iter=self.max_iter,
                                     n_jobs=self.n_jobs)



        if self.radius is None:
            self.distance_graph = kneighbors_graph(self.nbrs_, self.n_neighbors,
                                   mode='distance', n_jobs=self.n_jobs)
        else:
            self.distance_graph = radius_neighbors_graph(self.nbrs_, radius=self.radius,
                                                         mode='distance', n_jobs=self.n_jobs)

        if decompose_graph:
            temp = sparse.csgraph.connected_components(self.distance_graph)
            self.num_connected_comps = temp[0]
            self.connected_comps = []
            self.dist_matrices = []
            self.embeddings = []
            # self.distance_graphs = []
            for i0 in range(self.num_connected_comps):
                idx = temp[1] == i0
                graph = self.distance_graph[idx, :][:, idx]
                self.connected_comps.append(graph)
                dist_matrix = graph_shortest_path(graph,
                                           method=self.path_method,
                                           directed=False)
                self.dist_matrices.append(dist_matrix)
                G = dist_matrix ** 2
                G *= -0.5
                embedding = self.kernel_pca_.fit_transform(G)
                self.embeddings.append(embedding)
        else:
            self.dist_matrix_ = graph_shortest_path(self.distance_graph,
                                                    method=self.path_method,
                                                    directed=False)

            G = self.dist_matrix_ ** 2
            G *= -0.5

            self.embedding_ = self.kernel_pca_.fit_transform(G)

    def reconstruction_error(self):
        """Compute the reconstruction error for the embedding.

        Returns
        -------
        reconstruction_error : float

        Notes
        -----
        The cost function of an isomap embedding is

        ``E = frobenius_norm[K(D) - K(D_fit)] / n_samples``

        Where D is the matrix of distances for the input data X,
        D_fit is the matrix of distances for the output embedding X_fit,
        and K is the isomap kernel:

        ``K(D) = -0.5 * (I - 1/n_samples) * D^2 * (I - 1/n_samples)``
        """
        G = -0.5 * self.dist_matrix_ ** 2
        G_center = KernelCenterer().fit_transform(G)
        evals = self.kernel_pca_.lambdas_
        return np.sqrt(np.sum(G_center ** 2) - np.sum(evals ** 2)) / G.shape[0]

    def fit(self, X, y=None):
        """Compute the embedding vectors for data X

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree, NearestNeighbors}
            Sample data, shape = (n_samples, n_features), in the form of a
            numpy array, precomputed tree, or NearestNeighbors
            object.

        y : Ignored

        Returns
        -------
        self : returns an instance of self.
        """
        self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None, decompose_graph=False):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self._fit_transform(X, decompose_graph)
        if decompose_graph:
            return self.embeddings
        else:
            return self.embedding_


    def transform(self, X):
        """Transform X.

        This is implemented by linking the points X into the graph of geodesic
        distances of the training data. First the `n_neighbors` nearest
        neighbors of X are found in the training data, and from these the
        shortest geodesic distances from each point in X to each point in
        the training data are computed in order to construct the kernel.
        The embedding of X is the projection of this kernel onto the
        embedding vectors of the training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X = check_array(X)
        distances, indices = self.nbrs_.kneighbors(X, return_distance=True)

        # Create the graph of shortest distances from X to self.training_data_
        # via the nearest neighbors of X.
        # This can be done as a single array operation, but it potentially
        # takes a lot of memory.  To avoid that, use a loop:
        G_X = np.zeros((X.shape[0], self.training_data_.shape[0]))
        for i in range(X.shape[0]):
            G_X[i] = np.min(self.dist_matrix_[indices[i]] +
                            distances[i][:, None], 0)

        G_X **= 2
        G_X *= -0.5

        return self.kernel_pca_.transform(G_X)
