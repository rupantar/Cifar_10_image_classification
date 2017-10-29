##softmax naive implementation 
import numpy as numpy
from random import shuffle

def svm_loss_naive(W,X,y,reg=0.00):
	

	import numpy as np	

	#initialize loss and the gradient to zero
	delta = 1
	loss = 0.0
	D = np.zeros_like(W)
	scores = W.dot(X)
	correct_class_scores = scores[y]

	num_train = X.shape[0]
	num_classes = W.shape[1]

	for i in xrange(num_train):   #iterate throught all the wrong classes

		if i == j :

			#skip the loss for the true class to only loop over the incorrect classes
			continue

			# accumulate loss for the i-th example

		loss +=max(scores[i] - correct_class_scores +delta)

	loss /= num_train	
	return loss

def svm_loss_half_vectorized(W,X,y,reg=0.00):
	"""
	A faster implementation with half vectorization. For a single example this implementation has no for loops ,but there is 
	
	"""
	import numpy as np	

	delta = 1
	loss = 0.0
	scores = X.dot(W)
	num_train = X.shape[0]

	#compute the margins for all classes in one vector operation
	margins = np.maximum(0,scores - scores[y] + delta)

	margins[y] = 0

	loss = np.sum(margins)
	
	# as we want to calculate the average loss over all the training examples

	loss /=num_train
	
	return loss
