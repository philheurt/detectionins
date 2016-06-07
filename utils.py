import numpy as np
import string

def extract_features(filename, train=True):
	"""Récupère les commentaires ainsi que leur label si train=True (défaut)."""
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

def clean(stringtab):
	"""Enlève la ponctuation pour l'instant, peut-être plus à faire."""
	return [s.translate(str.maketrans("","", string.punctuation)) for s in stringtab]