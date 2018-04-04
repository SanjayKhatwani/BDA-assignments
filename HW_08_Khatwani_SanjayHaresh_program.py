import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def read_csv(filename):
    """
    This method reads a csv file row wise and returns the data in a list
    :param filename: name of csv file
    :return: list of rows in csv
    """
    data = []
    returnData = []
    with open(filename, 'r') as csvfile:
        recepiereader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in recepiereader:
            data.append(row)
        for row in range(1, len(data)):
            returnData.append([])
            for val in data[row]:
                returnData[row-1].append(int(val))
    return returnData

def computeCovariance(data):
    """
    This method uses the numpy to return covariance matrix of data
    :param data: data
    :return: Covariance matrix
    """
    values = [i[1:] for i in data]
    return np.cov(values, rowvar=False)

def compute_eigen(cov):
    """
    This method computes the eigen values and eigen vectors of covariances.
    Then sorts the eigen values in decreasing order along with corresponding
    eigen vectors.
    Then it normalizes the eigen values.
    And finally a graph of cumulative eigen values is plotted.
    :param cov: covariances
    :return: eigen values and eigen vectors
    """
    [w, v] = np.linalg.eig(cov)
    #Get sorting order
    idx = w.argsort()[::-1]
    #sort
    w = w[idx]
    v = v[:,idx]

    #Normalize
    valSum = sum(w)
    w = [i/valSum for i in w]

    w.insert(0, 0)

    #Calculate cumulative sum
    cumulativeSum = np.cumsum(w)
    plt.figure(1)
    plt.plot(cumulativeSum)
    plt.xlabel('Attributes')
    plt.ylabel('Cumulative Sum')
    plt.title('cumulative sum of normalized Eigen Values')
    plt.savefig('EigenSumSum.png')
    return [w, v]

def transform_to_2d(v, data):
    """
    This method projects the data on to 2-d space given by first 2 eigen
    vectors. Finally it plots a scatter gram of transformed data in that space.
    :param v: eigen vectors
    :param data: data
    :return: projected data
    """
    #Get first 2 eigen vectors
    matrix_vetors = np.hstack((v[:, 0].reshape(12, 1), v[:, 1].reshape(12, 1)))
    values = [i[1:] for i in data]

    #Project and plot
    transformed = matrix_vetors.T.dot(np.array(values).T)
    plt.figure(2)
    plt.scatter(transformed[0], transformed[1])
    plt.xlabel('Eigen-vector 1')
    plt.ylabel('Eigen-vector 2')
    plt.title('Original data transformed to 2D space')
    plt.savefig('TransformedTo2d.png')
    return transformed

def do_k_means(data, v):
    """
    This method performs k means on the projected data in 2-d space.
    Then it multiples the cluster centers to the corresponding eigen vectors.
    :param data: data
    :param v: eigen vectors
    :return: N/A
    """
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data.T)
    k_centers = kmeans.cluster_centers_
    print("K-Centers:")
    print(k_centers)

    #Get first 2 eigrn vectors.
    matrix_w = np.hstack((v[:, 0].reshape(12, 1), v[:, 1].reshape(12, 1)))

    #Multiple cluster centers with eigen vectorss.
    prototype_amts = k_centers.dot(matrix_w.T)
    print("Prototype-amounts: ")
    print(prototype_amts)

def main():
    """
    Main method
    :return:
    """
    data = read_csv('HW_AG_SHOPPING_CART_v5121.csv')
    w, v = compute_eigen(computeCovariance(data))
    print("First Eigen Vector: ")
    print(v[:,0])
    print("Second Eigen Vector: ")
    print(v[:,1])
    transformed_data = transform_to_2d(v, data)
    do_k_means(transformed_data, v)


if __name__ == '__main__':
    main()