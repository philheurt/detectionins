import numpy as np
import string 

def extract_features(filename, train=True):
	"""Recupere les commentaires ainsi que leur label si train=True (defaut)."""
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
	"""Enleve la ponctuation pour l'instant, peut-etre plus a faire."""
	empt = " "
	for i in range(31):
		empt += " "
	trantab = string.maketrans(string.punctuation, empt)
	return [s.translate(trantab) for s in stringtab]

def tokenise(stringtab):
	return [s.split(' ') for s in stringtab]