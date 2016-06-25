import numpy as np
class LogisticRegression:

	def __init__(self,tolerance=1e-6):
		"""Initializes Class for Logistic Regression
		
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
		self.likelihood_history = None
		self.labels = None
		self.w = None
		self.features = None
		self.X = None
		self.mean_x = None
		self.std_x = None

	def fit(self,X,y):
		""" Fit Class for Logistic Regression
		Parameters
		----------
		X : ndarray(n-rows,m-features)
			Numerical training data.		
		y: ndarray(n-rows,)
			Integer training labels.
		"""
		self.X = X
		self.labels = y.reshape(y.size,1)
		# On initialise w avec un w0 qui est egal a 0
		self.w = np.zeros((X.shape[1]+1,1))
		# On ajoute l'intercept pour w0 qui est un vecteur de 1
		self.features = np.ones((X.shape[0],X.shape[1]+1))
		self.features[:,1:] = X
		self.mean_x = X.mean(axis=0)
		self.std_x = X.std(axis=0)
		print ("Starting the gradient descent with the standard parameters")
		self.gradient_decent()
		print ("Finished the gradient descent")

	def probability(self):
		"""Computes the logistic probability of being a positive example
		
		Returns
		-------
		out : ndarray (1,)
			Probablity of being a positive example
		"""
		# Fonction sigmoide 
		return 1 / (1 + np.exp(-self.features.dot(self.w)))

	# Fonction que l'on recherche a minimiser
	def log_likelihood(self):
		"""Calculate the loglikelihood for the current set of weights and features.
 
		Returns
		-------
		out : float
		""" 
		p = self.probability()
		# On ajoute 10^-24 afin de s'assurer qu'on ne prend pas un log nul par approximation
		loglikelihood = self.labels * np.log(p + 1e-24) + (1 - self.labels) * np.log(1 - p + 1e-24)
		return -1 * loglikelihood.sum()

	# Calcul de son gradient
	def log_likelihood_gradient(self):
		"""Calculate the loglikelihood gradient for the current set of weights and features.
 
		Returns
		-------
		out : ndarray(n features, 1)
			gradient of the loglikelihood
		""" 
		error = self.labels - self.probability()
		product = error * self.features
		return product.sum(axis = 0).reshape(self.w.shape)

	def gradient_decent(self,alpha= 1e-6,max_iterations = 1e5):
		"""Runs the gradient decent algorithm
		
		Parameters
		----------
		alpha : float
			The learning rate for the algorithm
		max_iterations : int
			The maximum number of iterations allowed to run before the algorithm terminates
			
		"""
		previous_likelihood = self.log_likelihood()
		difference = self.tolerance + 1
		iteration = 0
		self.likelihood_history = [previous_likelihood]
		while (difference > self.tolerance) and (iteration < max_iterations):
			self.w = self.w + alpha * self.log_likelihood_gradient()
			temp = self.log_likelihood()
			difference = np.abs(temp - previous_likelihood)
			previous_likelihood = temp
			self.likelihood_history.append(previous_likelihood)
			iteration += 1
	
	def row_probability(self,row):
		"""Computes the logistic probability of being a positive example for a given row
		
		Parameters
		----------
		row : int
			Row from feature matrix with to calculate the probablity.
			
		Returns
		-------
		out : ndarray (1,)
			Probablity of being a positive example
		"""
		return 1 / (1 + np.exp(-self.features[row,:].dot(self.w)))
		
			
	def predict_probability(self,X):
		"""Computes the logistic probability of being a positive example
		
		Parameters
		----------
		X : ndarray (n-rows,n-features)
			Test data to score using the current weights
		Returns
		-------
		out : ndarray (1,)
			Probablity of being a positive example
		"""
		features = np.ones((X.shape[0],X.shape[1]+1))
		# On standardise les donnees pour reduire le temps de calcul
		features[:,1:] = (X-self.mean_x)/self.std_x
		return 1/(1+np.exp(-features.dot(self.w)))

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
		features = np.ones((X.shape[0],X.shape[1]+1))
		# On standardise les donnees pour reduire le temps de calcul
		features[:,1:] = (X-self.mean_x)/self.std_x
		prediction = 1/(1+np.exp(-features.dot(self.w)))
		result = np.zeros(len(prediction))
		for i in range(len(prediction)):
			if(prediction[i] > 0.6):
				result[i] = 1
		return result

	def score(self,X,y):
		"""Computes the labels predicted according to the mode
		
		Parameters
		----------
		X : ndarray (n-rows,n-features)
			Test data to score using the current weights
		Y : ndarray(n-rows,)
			Integer testlabels.
		
		Returns
		-------
		out : integer
			Score
		"""
		features = np.ones((X.shape[0],X.shape[1]+1))
		# On standardise les donnees pour reduire le temps de calcul
		features[:,1:] = (X-self.mean_x)/self.std_x
		prediction = 1/(1+np.exp(-features.dot(self.w)))
		result = np.zeros(len(prediction))
		for i in range(len(prediction)):
			if(prediction[i] > 0.6):
				result[i] = 1
		result.mean()
		diff = abs(result - y)
		return 1 - diff.mean()

	def get_coefficients(self):
		# A utiliser lorsque l'on standardise les donnees
		new_coef = self.w.T[0] / np.hstack((1,self.std_x))
		new_coef[0] = self.w.T[0][0] - (self.mean_x * self.w.T[0][1:] / self.std_x).sum()
		return new_coef

