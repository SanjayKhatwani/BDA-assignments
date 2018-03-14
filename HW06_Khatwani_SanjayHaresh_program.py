import csv
import math
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

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

def calculate_means(data):
    """
    This method calculates the mean of all attributes in the data
    :param data: data
    :return: List of means
    """
    means = []
    for i in range(len(data[0])):
        values = [data[j][i] for j in range(len(data))]
        means.append(sum(values) / len(values))
    return means

def calculate_standard_deviations(data, means):
    """
    This method calculates the standard-deviations of all attributes in the
    data.
    :param data: data
    :param means: means of all attributes
    :return: List of standard deviations
    """
    sds = []
    for i in range(len(data[0])):
        values = [data[j][i] for j in range(len(data))]
        sd = math.sqrt(sum([pow(x - means[i], 2) for x in values]) / len(
            values))
        sds.append(sd)
    return sds

def calculate_corelations(data, means, sds):
    """
    This method calculates the cross-correlation coefficients of all
    attributes with all other attributes
    :param data: data
    :param means: List of means
    :param sds: List of standard deviations
    :return: Cross correlation matrix
    """
    n = len(data)
    cc = []
    for attribute in range(1, len(means)):
        cc.append([])
        for attribute_other in range(1, len(means)):
            sum = 0
            for row in range(0, len(data)):
                sum += ((data[row][attribute] - means[attribute]) / sds[
                    attribute]) * ((data[row][attribute_other] - means[
                    attribute_other]) / sds[attribute_other])
            cc[attribute-1].append(round((sum / n), 2))
    return cc

def calculate_euclidean_distance(data1, data2):
    """
    This method calculates the euclidean distance between the two data points
    :param data1: data array 1
    :param data2: data array 2
    :return: Euclidean distance between data1 and data2
    """
    sum = 0
    for i in range(len(data1)):
        sum += pow((data2[i] - data1[i]), 2)
    return math.sqrt(sum)

def get_closest_clusters(centers):
    """
    This method finds the two clusters whoes centers are closest to one
    another. It implements central linkage.
    :param centers: List of center coordinates
    :return: indices of best clusters and distance between them
    """
    best_distance = 99999
    best_pair_1 = -1
    best_pair_2 = -1
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            if centers[i] != -1 and centers[j] != -1:
                dist = calculate_euclidean_distance(centers[i], centers[j])
                if dist < best_distance:
                    best_distance = dist
                    best_pair_1 = i
                    best_pair_2 = j
    return [[best_pair_1,best_pair_2], best_distance]

def get_center(data):
    """
    This method calculates the center of a cluster
    :param data: Cluster data
    :return: new center
    """
    center = []
    for j in range(len(data[0])):
        sum = 0
        for i in range(len(data)):
            sum += data[i][j]
        center.append(sum / len(data))
    return center

def merge_clusters_and_recalculate_center(pair, clusters, data, centers):
    """
    This method merges two clusters and reassigns their centers.
    :param pair: indices of clusters to merge
    :param clusters: all clusters
    :param data: data of all clusters
    :param centers: centers of all clusters
    :return: merged clusters and new centers
    """
    cluster_data = []
    for i in range(len(data)):
        if clusters[i] in pair:
            cluster_data.append(data[i])
    size_of_smaller_cluster = min(clusters.count(pair[0]), clusters.count(
        pair[1]))
    centers[max(pair)] = -1
    centers[min(pair)] = get_center(cluster_data)
    clusters = [min(pair) if x == max(pair) else x for x in clusters]

    return[clusters, centers, size_of_smaller_cluster]


def agglomerative_clustering(data, clusters, centers):
    """
    This method performs hierarchical clustering on data.
    :param data: data
    :param clusters: initial clusters where every point is its own cluster
    :param centers: same as data points initially.
    :return: clusters and their centers.
    """
    sizes = []
    while len(set(clusters)) > 1:
        closest_clusters, distance = get_closest_clusters(centers)

        clusters, centers, size_of_smaller_cluster = merge_clusters_and_recalculate_center(
            closest_clusters, clusters, data, centers)

        sizes.append(size_of_smaller_cluster)
    return [clusters, centers, sizes]


def main():
    """
    Main method
    :return: N/A.
    """
    data = read_csv('HW_AG_SHOPPING_CART_v512.csv')
    means = calculate_means(data)
    stddev = calculate_standard_deviations(data, means)
    coor = calculate_corelations(data, means, stddev)
    print("The cross-correlation coefficient matrix is: ")
    for row in coor:
        print(row)
    clusters = []
    data_no_id = []
    centers = []
    for row in data:
        clusters.append(row[0]-1)
        data_no_id.append(row[1:])
        centers.append(row[1:])
    clusters, centers, sizes = agglomerative_clustering(data_no_id, clusters,
                                                   centers)
    print(sizes)
    #Dendogram for central linkage
    Z = linkage(data_no_id, method='average')
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram - Central Linkage')
    plt.xlabel('Data Index')
    plt.ylabel('Distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.savefig("central.png")

    # Dendogram for central linkage
    Z = linkage(data_no_id, method='complete')
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram - Complete Linkage')
    plt.xlabel('Data Index')
    plt.ylabel('Distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.savefig("complete.png")

    # Dendogram for central linkage
    Z = linkage(data_no_id, method='single')
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram - Single Linkage')
    plt.xlabel('Data Index')
    plt.ylabel('Distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.savefig("single.png")

if __name__ == '__main__':
    main()