import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    N = labels.shape[0]

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    x = data
    a1 = x
    z1 = x
    z2 = np.dot(a1,W1) + b1
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W2) + b2

    a3 = softmax(z3) #normalized for probabilty
    #loss function would be here ? https://en.wikipedia.org/wiki/Cross_entropy or lecture 6
    cost = - np.sum(np.log(np.sum(labels * a3, axis=1))) / N #this labels * a3 multiplication only keeps the non 0 vector, we use the softmax function here

    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    error = (a3 - labels) / N
    
    gradW2 = np.dot(np.transpose(a2), error)
    gradb2 = np.sum(error, axis=0)

    delta2 = sigmoid_grad(a2) * np.dot(error, np.transpose(W2))
    gradW1 = np.dot( np.transpose(x), delta2)
    gradb1 = np.sum(delta2, axis=0)

    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
        
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20 #number of window to classify
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1 # give probability 1 to each row, give a 100% label to each row
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
