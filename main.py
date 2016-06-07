# coding: utf8

import numpy as np
import utils
import re

def main():
	# extraction du train
	X, y = utils.extract_features('train.csv')

	# extraction du test
	X_test = utils.extract_features('test.csv', train=False)

	# on retire la ponctuation
	X = utils.clean(X)
	X_test = utils.clean(X_test)


if __name__ == '__main__':
	main()
