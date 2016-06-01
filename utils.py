import numpy as np

def extract_features(filename, train=True):
	X=[]
	y=[]
	with open(filename) as f:
		if train:
			for line in f:
				y.append(int(line[0]))
				X.append(line[5:-4])

			y = np.array(y)
			return X, y
		else:
			for line in f:
				X.append(line[1:-1])
				
			return X