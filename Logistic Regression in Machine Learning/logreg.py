
#I wanted to import the numpy library to facilitate mathematical operations.
import numpy as np

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.theta = None
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        # I get the dimensions of the data
        number_of_rows = X.shape[0]

        # I calculate the values for the sigmoid function applied to X * theta
        retval_sigmoid = self.sigmoid(np.dot(X, theta))

        # First term: -y.T * np.log(retval_sigmoid)
        logarirthm_sigmoid = np.log(retval_sigmoid)
        first_term = -np.dot(y.T, logarirthm_sigmoid)

        # Second term: -(1.0 - y).T * np.log(1.0 - retval_sigmoid)
        logarithm_sigmoid2 = np.log(1.0 - retval_sigmoid)
        second_term = -np.dot((1.0 - y).T, logarithm_sigmoid2)

        # I divide the sum of the first and second terms by number_of_rows
        cost_without_regularization = (first_term + second_term) / number_of_rows

        # I calculate the regularization term: regLambda / (2 * number_of_rows) * (theta.T * theta)
        square_of_theta = np.dot(theta.T, theta)  # Square of theta
        regularization = (self.regLambda / (2.0 * number_of_rows)) * square_of_theta

        # I compute the final cost function by adding the cost terms and regularization term
        total_cost = cost_without_regularization + regularization

        # I return the total_cost as a scalar value
        return total_cost  # Directly returning scalar value


    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function using NumPy
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, a d-dimensional numpy vector
        '''
        # I get the number of rows (observations) in the dataset
        number_of_rows = X.shape[0]
        
        # Step 1: I compute the hypothesis (predicted values using the sigmoid function)
        hypothesis = self.sigmoid(np.dot(X, theta))  # I calculate the hypothesis
        
        # Step 2: I calculate the error between predicted and actual values
        error = hypothesis - y  # I store the difference between prediction and actual values
        
        # Step 3: I compute the unregularized gradient
        unregularized_gradient = np.dot(X.T, error) / number_of_rows  # I compute the unregularized part
        
        # Step 4: I add the regularization term (skip the first element for the bias term)
        regularization_term = (regLambda / number_of_rows) * theta  # I calculate the regularization component
        regularization_term[0] = 0  # I exclude the bias term from regularization
        
        # Step 5: I combine both parts
        gradient_value = unregularized_gradient + regularization_term  # I combine to get the total gradient
        
        # Step 6: I adjust the gradient for the bias term (theta[0])
        gradient_value[0] = np.sum(error) / number_of_rows  # I handle the first parameter separately
        
        return gradient_value


    #As you mentioned in the Homework2 pdf, I added the hasConverged function to this file.
    def hasConverged(self, theta_new, theta_old):
        if np.linalg.norm(theta_new - theta_old) < self.epsilon:
            return True
        else:
            return False


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        y = y.reshape(-1, 1)  # I reshape y to have the shape (n, 1)
        number_of_rows, number_of_columns = X.shape
        # I add the bias term (1's feature) to X
        X = np.c_[np.ones((number_of_rows, 1)), X]

        # I create a random starting theta as a numpy array (instead of np.mat)
        self.theta = np.random.rand(number_of_columns + 1, 1)  # np.mat yerine np.array

        old_theta_value = self.theta
        new_theta_value = self.theta

        i = 0
        while i < self.maxNumIters:
            # I compute the new theta using the gradient descent update rule
            new_theta_value = old_theta_value - self.alpha * self.computeGradient(new_theta_value, X, y, self.regLambda)

            # I check for convergence: if the change in theta is less than epsilon, I stop
            if self.hasConverged(new_theta_value, old_theta_value):
                self.theta = new_theta_value
                return
            else:
                # I update theta_old for the next iteration
                old_theta_value = np.copy(new_theta_value)
                i += 1
                # I compute the cost (optional for debugging)
                total_cost = self.computeCost(new_theta_value, X, y, self.regLambda)
                # print("Total Cost: ", total_cost)

        # After the loop, I set the final value of theta
        self.theta = new_theta_value


    def predict(self, X):
        '''
        Uses the trained model to make predictions for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector with the predictions
        '''
        number_of_rows = X.shape[0]  # I get the number of samples
        
        # I add a column of ones to X to include the bias term (theta_0)
        arrays_of_ones = np.c_[np.ones((number_of_rows, 1)), X]  # Using np.c_ to add bias term directly
        
        # I use the sigmoid function to calculate predictions (probabilities)
        predictions = self.sigmoid(np.dot(arrays_of_ones, self.theta))  # Compute the dot product of X and theta
        
        # Return binary predictions (0 or 1) based on a threshold of 0.5
        return (predictions >= 0.5).astype(int)




    def sigmoid(self, z):
        '''
        This method wasn't provided in the hw template...
        Computes sigmoid for both vectors and matrices
        '''
        # Sigmoid function: 1 / (1 + e^(-z))
        return 1.0 / (1.0 + np.exp(-z))