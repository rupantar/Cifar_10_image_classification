import numpy as np

class KNearestNeighbor(object):
    """A knn implementation with L2 distance."""
    
    def __init__(self):
        pass
    
    def train(self,training_data,training_labels):
        """
        Just remembers the training data
        Input: training data array and the Y-labels
        Does not return anything
        """
        
        self.X_train = training_data
        self.Y_train = training_labels
        
    def predict(self, test_data, k = 1, num_loops = 1):
        
        """
        Predict the labels for the test data using this classifier.
        Input: test data os a N*D dimensional array where each row is a test point. Test data is N dimensional.
        Output: y_predict is the predicted labels of data, where y[i] is the predicted label for the point test_data[i]
        """
        y_predict = np.zeros(test_data.shape[0],dtype = self.training_labels.dtype)
        
        if num_loops ==0:
            dists = self.compute_distance_using_no_loop(X)
        if num_loops ==1:
            dists = self.compute_distance_using_one_loop(X)
#        if num_loops ==2:
#            dists = self.compute_distance_using_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        
        y_predict = self.predict_labels(dists,k=k)
    
        return y_predict
    
    
    def compute_distance_using_two_loops(self,testing_data):
        #we will use l2 distance metric to calculate the distance
        """
        Compute the distance of each test point in testing_data and each training point stored in 
        self.X_train using two for loops.
        
        Input:
        
        Output: distance matrix with shape(num_test,num_train) where dist[i,j] is the euclidean distance between
        the ith test point and the jth training example.
        
        """
        num_tests = testing_data.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_tests,num_train))
        for i in xrange(num_tests):
            for j in xrange(num_train):
                
                dists[i,j] = np.sqrt(np.sum((testing_data[i,:]-self.X_train[j,:])**2))
        
        return dists
    
    def compute_distance_using_one_loop(self,testing_data):
        # same thing as before but using only one for loop 
        
        num_tests = testing_data.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_tests,num_train))
        
        for i in xrange(num_tests):
            
            dists[i,:] = np.sqrt(np.sum(np.square(testing_data[i,:]-self.X_train),axis = 1))
        
        return dists
    
    def predict_labels(self,dists,k=1):
        """
        Given a matrix of distances between test points and training points, 
        predict the label for each of the test point.
        
        Inputs:
        - dists - A numpy array of distance b/w the test points and the training examples
        
        Outputs: return an array with dimension [num_test,:] 
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        
        for i in xrange(num_test):
            # A list of length k storing the labels of the k nearest neighbors to the ith test point
            closest_y = []
            
            k_nearest_idx = np.argsort(dists[i,:],axis = 0)
            closest_y = self.Y_train[k_nearest_idx[:k]]
            
            y_pred[i] = np.argmax(np.bincount(closest_y))
            
        return y_pred
        
