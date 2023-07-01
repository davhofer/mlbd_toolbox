"""
K-means
spectral clustering

elbow method viz
silhouette score
BIC score, bayesian informatio ncriterion, schwarz

time series clustering, lecture 11
dynamic time warping
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import distance
from scipy.sparse.csgraph import laplacian
from scipy import linalg

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, rand_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import spectral_embedding


"""
make data:
X_blobs, y_blobs = make_blobs(n_samples=n_samples, centers=4,
                              cluster_std=0.7, random_state=2010)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], s=50, c=y_blobs)


clustering:
k_means = KMeans(n_clusters=k, max_iter=1, init=centroids, n_init=1)
y_hat = k_means.fit_predict(X)
c_hat = k_means.cluster_centers_

"""


def plot_distortion(n_clusters_list, X):
    """
    Plot the distortion (in-cluster variance) on the y-axis and 
    the number of clusters in the x-axis 
    
    :param n_clusters_list: List of number of clusters to explore
    :param X: np array of data points 
    """
    distortion_list = []
    for k in n_clusters_list:
        kmeans = KMeans(n_clusters=k, random_state=111).fit(X)
        distortion = kmeans.inertia_
        distortion_list.append(distortion)

    plt.plot(n_clusters_list, distortion_list, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    
    
# n_clusters_list = range(2, 10)    
# plot_distortion(n_clusters_list, X_blobs)

def plot_silhouette(n_clusters_list, X):
    """
    Plot the silhouette score on the y-axis and
    the number of clusters in the x-axis
    :param n_clusters_list: List of number of clusters to explore
    :param X: np array of data points 
    """
    silhouette_list = []
    for k in n_clusters_list:
        kmeans = KMeans(n_clusters=k, random_state=111)
        y_pred = kmeans.fit_predict(X)
        silhouette = silhouette_score(X, y_pred)
        silhouette_list.append(silhouette)

    plt.plot(n_clusters_list, silhouette_list, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette width')


def compute_bic(kmeans, X, clustering_method='kmeans'):
    """
    Computes the BIC metric

    :param kmeans: clustering object from scikit learn
    :param X: np array of data points
    :return: BIC
    """
    # Adapted from: https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans

    # number of clusters
    k = kmeans.n_clusters
    labels = kmeans.labels_
    
    if clustering_method=='spectral':
        centers = [np.array([np.mean(X[labels==i], axis=0) for i in range(k)])]
    else:
        centers = [kmeans.cluster_centers_]
        
    # size of the clusters
    n = np.bincount(labels)
    # size of data set
    N, D = X.shape

    # compute variance for all clusters beforehand
    cl_var = (1.0 / (N - k) / D) * sum([sum(distance.cdist(X[np.where(labels == i)], \
                                    [centers[0][i]],'euclidean') ** 2) for i in range(k)])


    LL = np.sum([n[i] * np.log(n[i]) -
                  n[i] * np.log(N) -
                  ((n[i] * D) / 2) * np.log(2 * np.pi * cl_var) -
                  ((D / 2)*(n[i] - 1))  for i in range(k)])
    
    d = (k - 1) + 1 + k * D
    const_term = (d / 2) * np.log(N)
    
    BIC = LL - const_term
    
    return BIC


# kmeans = KMeans(n_clusters=4, random_state=111).fit(X_blobs)
# bic = compute_bic(kmeans, X_blobs)

def plot_bic(n_clusters_list, X):
    """
    Plot the BIC on the y-axis and the number of clusters in the x-axis
    :param n_clusters_list: List of number of clusters to explore
    :param X: np array of data points 
    """
    bic_list = []
    for k in n_clusters_list:
        kmeans = KMeans(n_clusters=k, random_state=111).fit(X)
        bic = compute_bic(kmeans, X)
        bic_list.append(bic)

    plt.plot(n_clusters_list, bic_list, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('BIC')


# plot_bic(n_clusters_list, X_blobs)


"""
Spectral clustering
"""
def get_similarity(X, gamma = 1):
    """
    Computes the similarity matrix
    :param X: np array of data
    :param gamma: the width of the kernel
    :return: similarity matrix
    """
    
    similarity = pairwise_kernels(X, metric='rbf', gamma=gamma)
    
    return similarity


def get_adjacency(S, connectivity='full'):
    """
    Computes the adjacency matrix
    :param S: np array of similarity matrix
    :param connectivity: type of connectivity 
    :return: adjacency matrix
    """
    
    if(connectivity=='full'):
        adjacency = S
    elif(connectivity=='epsilon'):
        epsilon = 0.5
        adjacency = np.where(S > epsilon, 1, 0)
    else:
        raise RuntimeError('Method not supported')
        
    return adjacency


def spectral_clustering(W, n_clusters, random_state=111):
    """
    Spectral clustering
    :param W: np array of adjacency matrix
    :param n_clusters: number of clusters
    :param normed: normalized or unnormalized Laplacian
    :return: tuple (kmeans, proj_X, eigenvals_sorted)
        WHERE
        kmeans scikit learn clustering object
        proj_X is np array of transformed data points
        eigenvals_sorted is np array with ordered eigenvalues 
        
    """
    # Compute eigengap heuristic
    L = laplacian(W, normed=True)
    eigenvals, _ = linalg.eig(L)
    eigenvals = np.real(eigenvals)
    eigenvals_sorted = eigenvals[np.argsort(eigenvals)]

    # Create embedding
    random_state = np.random.RandomState(random_state)
    proj_X = spectral_embedding(W, n_components=n_clusters,
                              random_state=random_state,
                              drop_first=False)

    # Cluster the points using k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state = random_state)
    kmeans.fit(proj_X)

    return kmeans, proj_X, eigenvals_sorted


"""
n_clusters=4
random_state=0
data = X_blobs

labels_sc = SpectralClustering(n_clusters=n_clusters,
         random_state=random_state).fit_predict(data)

gamma = 1
S = get_similarity(data, gamma)
W = get_adjacency(S)
kmeans, proj_X, eigenvals_sorted = spectral_clustering(W, n_clusters, random_state=random_state)
labels_us = kmeans.labels_

rand_score(labels_us, labels_sc)
"""
