import numpy as np

def extract_features(filename, train=True):
	X=[]
	y=[]
	with open(filename) as f:
		for line in f:
			if train:
				y.append(int(line[0]))
				X.append(line[5:-4])
				return X, y
			else:
				X.append(line[1:-1])
				return X
	
	y = np.array(y)