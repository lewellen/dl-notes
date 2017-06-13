import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats

class DescentOptimizer:
	def __init__(self, step = 0.1, epsilon = 1e-6, maxIterations = 100):
		self.step = step
		self.epsilon = epsilon
		self.maxIterations = maxIterations

	def analyticGradientDescent(self, theta, dfd0):
		trace = []
		for _ in xrange(0, self.maxIterations):
			thetaP = theta
			trace.append(thetaP)

			grad = dfd0(theta)
			theta = np.add(theta, - self.step * grad)
			diff = np.linalg.norm(np.subtract(thetaP, theta))

			if(diff < self.epsilon):
				break

		return (theta, trace)

	def stochasticGradientDescent(self, theta, dfd0, X):
		sampleSize = int(0.2 * len(X))

		trace = []
		for _ in xrange(0, self.maxIterations):
			thetaP = theta
			trace.append(thetaP)

			grad = dfd0(theta, random.sample(X, sampleSize))
			theta = np.add(theta, - self.step * grad)
			diff = np.linalg.norm(np.subtract(thetaP, theta))

			if(diff < self.epsilon):
				break

		return (theta, trace)

def gradientDescentTest():
	# Zero finding / minimum finding / convex opt
	opt = DescentOptimizer()
	f = lambda x: 3 * np.power(x[0] + 1, 2) + np.power(x[1] - 1, 2)
	dfd0 = lambda x: np.array([
		3 * 2 * (x[0] + 1),
		2 * (x[1] - 1)
		])

	theta, trace = opt.fit(np.random.rand(2), f, dfd0)
	X = np.linspace(-1-1, -1+1, 100)
	Y = np.linspace(+1-1, +1+1, 100)

	Z = []
	for x in X:
		for y in Y:
			Z.append(f(np.array([x, y])))

	Z = np.array(Z).reshape((100, 100))

	fig, ax = plt.subplots()
	ax.contourf(X, Y, Z)
	ax.plot( map(lambda x : x[0], trace), map(lambda x : x[1], trace) )
	ax.scatter(theta[0], theta[1])
	plt.show()

def stochastGradientDescentTest():
	# Linear regression
	f = lambda x : 3 * x + 4

	xs = np.random.rand(30)
	ys = [f(x) + 0.5* (0.5 * random.random() - 1) for x in xs]

	# ax + by + c = (a, b, c) (x, y, 1)^T
	# y = a x + b

	stepx = np.array([0.1, 0])
	stepy = np.array([0, 0.1])

	g = lambda theta, x : np.dot(theta, [x, 1])

	J = lambda theta, xys : sum(np.power(np.dot(theta, np.array([x, 1])) - y, 2) for (x, y) in xys)
	dJd0 = lambda theta, xys : np.array([
		(J(theta + stepx, xys) - J(theta - stepx, xys)) / 2.0,
		(J(theta + stepy, xys) - J(theta - stepy, xys)) / 2.0
	])

	opt = DescentOptimizer()
	theta, trace = opt.stochasticGradientDescent(np.array(np.random.rand(2)), dJd0, zip(xs, ys))
	print theta

	X = np.linspace(0, 1, 100)
	Y = [g(theta, x) for x in X]
	fig, ax = plt.subplots()
	ax.plot(X, Y)
	ax.scatter(xs, ys)
	plt.show()

if __name__ == "__main__":
	# logistic regression
	logistic = lambda x: 1.0 / (1.0 + np.exp(-x))
	logisticModel = lambda w, x : logistic(np.dot(w, np.append(x, 1)))

	# cross entropy
	H = lambda t, p : t * np.log(p) + (1 - t) * np.log(1 - p)

	# Loss function
	L = lambda true, predicted : np.sum( H(t, p)  for (t, p) in zip(true, predicted) ) / float(len(predicted))

	J = lambda w, xys : -L( map(lambda xy : xy[1], xys), map(lambda xy: logisticModel(w, xy[0]), xys) )

	stepx = np.array([0.1, 0])
	stepy = np.array([0, 0.1])

	dJd0 = lambda theta, xys : np.array([
		(J(theta + stepx, xys) - J(theta - stepx, xys)) / 2.0,
		(J(theta + stepy, xys) - J(theta - stepy, xys)) / 2.0
	])

	muA = -3
	muB = +3
	sigmaA = 2
	sigmaB = 2

	xsA = scipy.stats.norm.rvs(muA, sigmaA, 100)
	ysA = np.repeat(-1, 100)

	xsB = scipy.stats.norm.rvs(muB, sigmaB, 100)
	ysB = np.repeat(+1, 100)

	xs = np.append(xsA, xsB)
	ys = np.append(ysA, ysB)

	opt = DescentOptimizer()
	theta, trace = opt.stochasticGradientDescent(np.array(np.random.rand(2)), dJd0, zip(xs, ys))
	print theta

	x0 = -theta[1] / theta[0]
	print x0

	support = np.linspace(-10, +10, 200)
	zeros = np.zeros(100)

	fig, ax = plt.subplots()
	ax.plot(support, scipy.stats.norm.pdf(support, muA, sigmaA))
	ax.scatter(xsA, zeros, marker='x')

	ax.plot(support, scipy.stats.norm.pdf(support, muB, sigmaB))
	ax.scatter(xsB, zeros, marker='x')

	ax.plot(support, map(lambda x : logisticModel(theta, x), support))

	ax.plot([x0, x0], [0, 1])
	
	plt.show()
