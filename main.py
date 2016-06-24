# coding: utf8

import numpy as np
import pandas as pd

import utils
import nlp

from NaiveBayesClassifier import NBC
from cross_val import cv_score

def main():

	X, y = utils.load_comments('train.csv')
	X_test = utils.load_comments('test.csv', test=True)

	X, X_test = utils.process(X, False), utils.process(X_test, False)

	clf = NBC()
	clf.fit(X,y)
	pred = clf.predict(X_test)

	np.savetxt('pred.txt', pred, fmt='%s')

if __name__ == '__main__':
	main()