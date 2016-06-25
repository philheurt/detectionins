import numpy as np


"""def gradient_descent(g, P0, gamma, epsilon):
    val, grad=g(P0)
    err = np.sum(grad**2)
    P=P0
    while err > epsilon:
        P=P-gamma*grad
        err = np.sum(grad**2)
        val, grad=g(P)
        
    return P, g(P)"""

def gradient_descent(X, y, gradient, w0_init, w_init, gamma, epsilon):

	grad = gradient(X, y, w0_init, w_init)
	err = np.sum(grad**2)
	vec = np.hstack([w0_init, w_init])

	while err > epsilon:
		vec = vec - gamma*grad
		grad = gradient(X, y, vec[0], vec[1:])
		err = np.sum(grad**2)

	return vec[0], vec[1:]


def armijos_descent(X, y, func, gradient, w0_init, w_init, a, b, beta, epsilon):

	val = func(X, y, w0_init, w_init)
	grad = gradient(X, y, w0_init, w_init)
	err = np.sum(grad**2)
	vec = np.hstack([w0_init, w_init])
	
	while err > epsilon:
		gamma = b
		while func(X, y, vec[0] - gamma*grad[0], vec[1:] - gamma*grad[1:]) > val - beta*gamma*np.sum(grad**2):
			gamma = gamma*a
		b = 2*gamma

		vec = vec - gamma*grad
		w0 = vec[0]
		w = vec[1:]
		val = func(X, y, w0, w)
		grad = gradient(X, y, w0, w)
		err = np.sum(grad**2)

	return vec[0], vec[1:]


"""def conjugate_gradient_descent(g, P0, a, b, beta, epsilon):
    d= -g(P0)[1]
    grad = g(P0) [0]
    err = np.sum(grad**2)
    P=P0
    while err > epsilon:
        # on choisit le pas optimal avec une line search
        gamma = b
        while g(P - gamma*grad)[0] > val - beta*gamma*np.sum(grad**2):
            gamma = gamma*a
            
        newP = P + gamma*d
        newgrad = g(newP)[1]
        c = np.sum(newgrad**2)/np.sum(grad**2)
        d = -newgrad + c*d
        P, grad = newP, newgrad
        err = np.sum(grad**2)
        
    return P, g(P)"""