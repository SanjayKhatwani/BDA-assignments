import csv
import os

def read_csv(filename):
    """
    This method reads a csv file row wise and returns the data in a list
    :param filename: name of csv file
    :return: list of rows in csv
    """
    data = []
    with open(filename, 'r') as csvfile:
        recepiereader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in recepiereader:
            data.append(row)
    return data


def segregate_data(data):
    """
    This method seperates the data into three sub lists: attribute names,
    values of all attributes in every row, class label of every row
    :param data: data list
    :return: [clazz, attribute, values]
    """
    attribute = data[0]
    clazz = []
    values = []
    for row in range(1, len(data)):
        if data[row][0] == "Cupcake":
            clazz.append(0)
        else:
            clazz.append(1)
        values.append([])
        for col in range(1, len(data[row])):
            values[row-1].append(float(data[row][col]))
    return [clazz, attribute, values]


def gini_index(partitions, classes, class_labels):
    """
    This method calculates the weighted gini index for given partitions
    :param partitions: partitions
    :param classes: partitioned class labels
    :param class_labels: List of all class labels
    :return:
    """
    n_instances = float(sum([len(group) for group in partitions]))
    gini = 0.0
    for part_index in range(len(partitions)):
        size = float(len(partitions[part_index]))  #Total number of rows
        if size == 0:
            continue
        score = 0.0
        #For every class....
        for class_val in class_labels:
            #Count number of rows and divide it by total size
            p = classes[part_index].count(class_val) / size
            #Summation
            score += p * p
        gini += (1.0 - score) * (size / n_instances)    #Weighted gini
    return gini


def emit_header(filename):
    """
    This method writes the header part to the classifier file.
    :param filename: name of classifier file
    :return: n/a
    """
    with open("Header.txt", 'r') as header_file:
        with open(filename, 'a') as classifier_file:
            classifier_file.write(header_file.read())


def split_data(attribute, value, data, clazz):
    """
    This method splits the given data on a given attribute and value.
    It also splits the list of class labels
    :param attribute: Index of attribute
    :param value: Value of the attribute to split on
    :param data: data values
    :param clazz: List of class labels of every row
    :return: partitioned data and class labels
    """
    left_split = []
    right_split = []
    left_split_class = []
    right_split_class = []
    for one_row in range(len(data)):
        if data[one_row][attribute] < value:
            left_split.append(data[one_row])
            left_split_class.append(clazz[one_row])
        else:
            right_split.append(data[one_row])
            right_split_class.append(clazz[one_row])
    return [[left_split, right_split], [left_split_class, right_split_class]]

def frange(start, end, step):
    """
    This function returns a list of range for float values
    :param start: range start
    :param end: range end
    :param step: step size
    :return:
    """
    tmp = start
    while(tmp < end):
        yield round(tmp, 1)
        tmp += step

def get_best_split(data, clazz):
    """
    This method tries every possible split for every attribute and returns
    the data split according to the best split (Lowest weighted gini).
    :param data: data values
    :param clazz: class labels
    :return: Dictionary with the split data, attribute it was split on and
    the value of that attribute
    """
    best_index = 999
    best_value = 999
    best_gini = 999
    best_split = None

    #For every attribute....
    for index in range(len(data[0]) - 1):
        min_v = min([item[index] for item in data])
        max_v = max([item[index] for item in data])
        #For every value....
        for value in frange(min_v, max_v, 0.1):
            groups, classes = split_data(index, value, data, clazz)
            gini = gini_index(groups, classes, [0, 1])
            if gini < best_gini:
                best_index = index
                best_value = value
                best_gini = gini
                best_split = [groups, classes]

    return {'partitions': best_split, 'attribute_index': best_index,
            'attribute_value': best_value}

def determine_class_of_node(clazz):
    """
    This method finds the decision of a node by finding the maximum number of
    class labels in the data in the node.
    :param clazz: List of class labels in the node.
    :return: decision of the node.
    """
    return max(set(clazz), key=clazz.count)

def expand_node(node):
    """
    This method splits the node into left and right nodes.
    :param node: node to split
    :return: node after spliting with left and right children.
    """
    #Get best split
    root = get_best_split(node['data'], node['clazz'])
    [[left, right], [left_clazz, right_clazz]] = root['partitions']
    del(root['partitions'])
    node['left'] = {'data':left, 'clazz':left_clazz}
    node['right'] = {'data':right, 'clazz':right_clazz}
    node['attribute_index'] = root['attribute_index']
    node['attribute_value'] = root['attribute_value']
    return node

def build_tree(data, clazz):
    """
    This method builds the tree node by node in a BFS fashion untill 98%
    accuracy is reached. It only builds 8 nodes in the tree.
    :param data: data values
    :param clazz: class labels
    :return: root of the decision tree
    """
    count = 1
    node = {'data':data, 'clazz':clazz}
    queue= []
    queue.append(node)

    while count <= 8 and len(queue) > 0:
        next = queue.pop(0)
        # Check for 98% accuracy. Since class = 1 or 0. Sum(class labels) =
        # 0.98*len if 98% of data is of class 1 and sum = 0.02*len if 98% of
        # data is of class 0
        if sum(next['clazz']) < 0.98 * len(next['clazz']) or\
            sum(next['clazz']) > 0.02 * len(next['clazz']):
            nextNode = expand_node(next)
            count += 1
            queue.append(nextNode['left'])
            queue.append(nextNode['right'])
    return node

def emit_classifier(root, depth, filename):
    """
    This method writes deduce method code into classifier by converting
    decision tree into if statements.
    This is a recursive method that processes nodes in a DFS fashion.
    :param root: root of decision tree
    :param depth: depth of the node
    :param filename: name of classifier file
    :return: n/a
    """
    has_left = 'left' in root.keys()
    has_right = 'right' in root.keys()

    #If it is a decision node, we need to print if else statements
    if has_left or has_right:
        #Build if statement
        string = '\n%s%s' % ((depth * '\t', 'if float(data[' + str(root[
                'attribute_index']) + ']) < ' + \
                 str(root['attribute_value']) + ':'))
        with open(filename, 'a') as classifier_file:
            classifier_file.write(string)
        if has_left:
            #Call on left child
            emit_classifier(root['left'], depth+1, filename)
        #Write else for right child
        string = '\n%s%s' % ((depth * '\t', 'else:'))
        with open(filename, 'a') as classifier_file:
            classifier_file.write(string)
        if has_right:
            #Call in right child
            emit_classifier(root['right'], depth+1, filename)
    #If it is a leaf node, we need to print return statements
    else:
        string = '\n%s%s' % ((depth * '\t', 'return ' + str(
            determine_class_of_node(root['clazz']))))
        with open(filename, 'a') as classifier_file:
            classifier_file.write(string)


def emit_trailer(filename):
    """
    This method writes the final trailing code into the classifier file.
    :param filename: name of classifier file.
    :return: n/a
    """
    with open("Trailer.txt", 'r') as header_file:
        with open(filename, 'a') as classifier_file:
            classifier_file.write("\n")
            classifier_file.write(header_file.read())

def main():
    """
    Main method
    :return: n/a
    """
    data = read_csv('Recipes_For_Release_2175_v201.csv')
    clazz, attribute, values = segregate_data(data)
    filename = "HW_06_Khatwani_SanjayHaresh_Classifier.py"
    #Remove file if existing
    try:
        os.remove(filename)
    except OSError:
        pass
    emit_header(filename)
    #Build decision tree.
    tree = build_tree(values, clazz)
    emit_classifier(tree, 1, filename)
    emit_trailer(filename)


if __name__ == '__main__':
    main()
