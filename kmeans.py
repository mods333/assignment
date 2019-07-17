import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

class Kmeans():

    def __init__(self, num_clusters:int = 10, max_iterations:int = 100000, tol:float= 1e-10) -> None:

        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.tol = tol

    def fit(self, data):

        assert data.shape[1] == 2
        cluster_centers = data[np.random.randint(data.shape[0], size=self.num_clusters)]
        for j in range(self.max_iterations):
            distances = cdist(data, cluster_centers)
            data_cluster = np.argmin(distances, axis=1)
            new_centers = np.zeros_like(cluster_centers)

            for i in range(self.num_clusters):
                new_centers[i,:] = np.mean(data[data_cluster==i], axis=0)

            dist = np.sqrt(np.sum(new_centers - cluster_centers)**2)
            cluster_centers = new_centers
            if dist < self.tol:
                print("convergef before max iterations at:{}".format(j))
                break
        return data_cluster, cluster_centers



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='K means clustering.')
    parser.add_argument('--num-data-points', help="Number of data points in the dataset", type=int, default= 1000)
    parser.add_argument('--num-blobs', help="Number of blobs in the dataset", type=int, default=10)
    parser.add_argument('--k', help="Number of cluster centroids for k means", type=int, default=10)
    parser.add_argument('--max-iter', help="Maximum number of iteration for k means", type=int, default=10000)
    args = parser.parse_args()

    #create test data
    data, target = make_blobs(n_samples = args.num_data_points,
                  n_features = 2,
                  centers = args.num_blobs,
                  cluster_std = 0.55,
                  shuffle = True)

    #plotting the original clusters

    plt.scatter(data[:,0], data[:,1], c = target, alpha=0.5)
    plt.title('Original data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    model = Kmeans(num_clusters=args.k, max_iterations=args.max_iter)
    out, cluster_centers = model.fit(data)

    plt.scatter(data[:,0], data[:,1], c = out, alpha=0.5)
    plt.scatter(cluster_centers[:,0], cluster_centers[:,1])
    plt.title('Learned clusters')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
