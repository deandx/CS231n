from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1] #(3073, 10)
    num_train = X.shape[0] #(500, 3073)
    
    for i in range(num_train):
        scores = X[i].dot(W) #(10,1)
        scores -= np.max(scores)
        loss += np.log(np.sum(np.exp(scores))) - scores[y[i]]
        
        exp_sum = np.sum(np.exp(scores))
        
        for j in range(num_classes):
            if j == y[i]:
                dW[:,j] -= X[i]
            dW[:, j] += np.exp(scores[j]) / exp_sum * X[i]  
        
    loss = loss / num_train + 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1] #(3073, 10)
    num_train = X.shape[0] #(500, 3073)
    
    scores = X.dot(W) #(500, 10)
    
    scores -= scores.max(axis = 1).reshape(-1, 1)
    
    loss = np.sum(np.log(np.sum(np.exp(scores), axis = 1)) - scores[range(num_train),y])
    
    exp_sum = np.sum(np.exp(scores), axis = 1).reshape(-1, 1)
    
    loss = loss / num_train + 0.5 * reg * np.sum(W * W)
    
    counter = np.zeros_like(scores)
    
    counter[range(num_train), y] -= 1
    
    dW += np.dot(X.T , np.exp(scores) / exp_sum + counter)
    
    dW = dW / num_train + reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
