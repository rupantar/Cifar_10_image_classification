import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.sparse


def getLoss(W,X,y,reg):
	
	# W is DxK matrix 
	# X is NxD matrix
	 
	# our scores matrix will be X.W which is NxK matrix

	 num_train = X.shape[0]
	 num_classes =  W.shape[1]

	 scores = X.dot(W)
	
	 y_ohe = one_hot_end(y)

	 probs = softmax(scores)

	 # remember svm loss is sigma(y*logp)
	
	 # log_probabs = -plog(p)
	 
	 correct_log_probs = y_ohe * np.log(probs) * (-1)
	#TODO:is loss just the sum of the correct log probs or the sum of the difference between the probs and 1 
         cross_entropy_loss = crossEntropyLoss(correct_log_probs)
	# reg_loss = 0.5*reg*np.sum(W*W)	 
	
	 total_loss = cross_entropy_loss #+ reg_loss
	#gradient update = -1/m * X.(y-scores)
	
	 grad = (-1/num_train) * np.dot(X.T,(y_ohe - scores))

	 return  cross_entropy_loss, grad

def crossEntropyLoss(individual_loss):

	num_train = individual_loss.shape[0]
	
	crossEntropyLoss = np.sum(individual_loss)/num_train
	
	return crossEntropyLoss 

def one_hot_end(y):
	m = y.shape[0]

	OHX = scipy.sparse.csr_matrix((np.ones(m),(y,np.array(range(m)))))

	OHX = np.array(OHX.todense()).T

	return OHX 


def softmax(scores):
	# Remember scores has a dimention of NxK matrix
	scores -= np.max(scores)#for numerical stability

	# we transpose the exp as it can be divided by the broadcasts	
	# Axis=1 makes it sum across the rows 
	probs = (np.exp(scores).T/np.sum(np.exp(scores),axis = 1)).T 

	return probs


def getProbsAndPreds(someX):

	probs = softmax(np.dot(someX,W))
	preds = np.argmax(probs,axis=1)

	return probs, preds

