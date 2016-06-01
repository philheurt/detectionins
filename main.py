import numpy as np
import matplotlib.pyplot as plt
import utils

def main():
	# extraction du train
	X, y = utils.extract_features('train.csv')

	# extraction du test
	X_test = utils.extract_features('test.csv', train=False)

	# nettoyer les données : enlever ponctuation, tokeniser

if __name__ == '__main__':
	main()
