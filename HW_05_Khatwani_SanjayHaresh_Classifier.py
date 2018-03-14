import csv
def read_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        recepiereader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in recepiereader:
            data.append(row[1:])
    return data[1:]

def deduce(data):
	if float(data[2]) < 19.4:
		if float(data[4]) < 11.3:
			if float(data[3]) < 17.1:
				if float(data[5]) < 2.1:
					return 1
				else:
					return 0
			else:
				return 0
		else:
			if float(data[0]) < 37.6:
				return 0
			else:
				return 1
	else:
		if float(data[4]) < 20.0:
			if float(data[0]) < 40.7:
				return 0
			else:
				return 0
		else:
			if float(data[0]) < 38.4:
				return 0
			else:
				return 1

def classify(data):
    classification = []
    for row in data:
        classification.append(deduce(row))
    return classification

def main():
    data = read_csv('Recipes_For_VALIDATION_2175_RELEASED_v201.csv')
    rows = classify(data)
    with open('validation_results.csv', "w", newline='') as results_file:
        writer = csv.writer(results_file)
        for row in rows:
            writer.writerow([row])

if __name__ == '__main__':
    main()