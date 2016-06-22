# coding: utf8

import numpy as np
import pandas as pd

import utils
import nlp

def main():
	# extraction des données
	X, y = utils.load_comments('train.csv')
	X_test = utils.load_comments('test.csv', test=True)

	#corp = utils.clean(X)
	#corp = utils.tokenize(corp)
	#corp = utils.remove_stop_words_punctuation(corp)
	#corp = nlp.stem(corp)

	#n = len(corp)

	#vocab = utils.get_vocab(corp)
	#print(len(vocab))
	df = utils.get_features(X)
	data = df.as_matrix()
	print(data[2764,:].max())
	print(data.shape)

if __name__ == '__main__':
	main()