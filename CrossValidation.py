import numpy as np
import LogisticRegression as lr

class CrossValidation:

	def __init__(self,X,y,k=10,tolerance=1e-5):
		"""Initializes Class for Cross Validation
			
		Parameters
		----------
		X : ndarray(n-rows,m-features)
			Numerical training data.		
		y: ndarray(n-rows,)
			Integer training labels.
				
		tolerance : float (default 1e-5)
			Stopping threshold difference in the loglikelihood between iterations.
				
		"""
		self.tolerance = tolerance
		self.X = X
		self.y = y
		self.k = k
		self.splits =  np.array_split(X,k)
		self.ysplits = np.array_split(y,k)

	def predict(self,X):
		"""Computes the labels predicted according to the mode
			
		Parameters
		----------
		X : ndarray (n-rows,n-features)
			Test data to score using the current weights
		Returns
		-------
		out : ndarray (1,)
			Labels
		"""
		somme = np.zeros((X.shape[0],1))
		#print somme

		for i in range(len(self.splits)):
			logreg = lr.LogisticRegression(self.splits[i],self.ysplits[i],tolerance=1e-6)
			logreg.gradient_decent(alpha=1e-2,max_iterations=1e4)
			somme = somme + logreg.predict_probability(X)
			#print logreg.predict_probability(X).shape
		somme = somme / self.k
		result = np.zeros(len(somme))
		for i in range(len(somme)):
			if(somme[i] > 0.5):
				result[i] = 1
		return result




				

			