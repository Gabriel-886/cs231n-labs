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
    S = X.dot(W)

    dLscore = np.zeros_like(S)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
        score = X[i].dot(W)
        score -= np.max(score)
        correct_score = score[y[i]]
        exp_sum = np.sum(np.exp(score))
        loss += np.log(exp_sum) - correct_score
        dLscore[i,y[i]] = -1
        for j in range(num_class):
            dLscore[i,j] += np.exp(score[j])/exp_sum
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW = X.T.dot(dLscore)
    loss = loss/num_train + 0.5 * reg * np.sum(W*W)
    dW = dW/num_train + reg * W
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

    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W) 
    scores -= np.max(scores, axis = 1).reshape(-1, 1)
    scores = np.exp(scores)
    sum_scores = np.sum(scores, axis = 1).reshape(-1, 1)
    class_prob = scores / sum_scores
    L_i = class_prob[range(num_train), y]
    loss = np.sum(-np.log(L_i)) / num_train
    loss += reg * np.sum(W * W)
    
    class_prob[range(num_train), y] -= 1
    dW = X.T.dot(class_prob)
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
