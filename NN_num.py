#Lesson1
from typing import Match
import numpy as np
import matplotlib
import matplotlib.pyplot as plt # for plotting
import itertools

from numpy.core.fromnumeric import reshape

def apply_layer (y_in = [], w = [], b = [], nlf = 0):
    '''
    y_in = previous layer results, i.e., a vector 1xN0
    b = next layer bias, i.e., a vector 1xN
    w = the connection weights, i.e., a NxN0 matrix 
    '''
    z = np.dot(y_in,w.T) + b
    if (nlf == 0) :
        y_out = sigmoid(z)
    else: y_out = np.array(np.heaviside(z,0))

    return y_out

def sigmoid (z=0):
    return 1/(1 + np.exp(-z))

def create_RandomNet (Ns =[]):
    ''' creates a random neural net
    '''
    # defining layers:
    net = []
    #loop for generating layers
    for j in range(len(Ns)):
        if j +1 < len(Ns): 
            w = np.random.uniform(low=-10,high=10,size=(Ns[j+1],Ns[j]))
            b = np.random.uniform(low=-1,high=1,size=(Ns[j+1]))
            net.append([w,b])
        #else: 
        #    b = np.random.uniform(low=-1,high=1,size=(Ns[j]))
        #    net.append([0,b])
    return net

def apply_net (net = [] ,y_in =[], nlf=0):
    ''' net must be a list of arrays, each item j being 
    [w,b] = [weights matrix, bias vector]
    Function: Loops the layers
    '''
    for j in range(len(net)):
        if j < len(net):
            y_in = apply_layer(y_in,net[j][0],net[j][1], nlf)
        else: y_in = y_in + net[j][1]
    return y_in

def Plot_2Dnetwork2(nlf=0, M=800):
    global net
    v0,v1 = np.meshgrid(np.linspace(-0.5,0.5,M),np.linspace(-0.5,0.5,M))
    v0 = v0.flatten()
    v1 = v1.flatten()
    y_in = np.zeros([np.shape(v0)[0],2])
    y_in[:,0] = v0
    y_in[:,1] = v1
    y_out = apply_net(net, y_in, nlf)
    y_plot = reshape(y_out[:,0], [M,M]) 
    plt.imshow(y_plot, origin='lower')
    plt.title(f"MxM output")
    plt.show()
    return 0

if __name__ == "__main__":
    # Running a random net
    '''
    Ns = [2,10, 33, 170,21,120, 88, 22,1] #nodes per layer
    net = create_RandomNet(Ns) #calls for a random net

    #running one time
    y_in = np.array([0.2,0.7]) # choose input
    y_out = apply_net(net, y_in, nlf =0)
    
    #plotting: 
    Plot_2Dnetwork2(nlf = 0, M=400) # running all values between -1 and 1
    '''

    # XOR gate with 3 layers: 
    ''' 
    y1,y2 <0 then answer is 0, otherwise its 1
    take y randomly 
    1st layer: do nothing and apply step function as your non-linear function
    2nd layer: take y1 - y2 and apply step function
    '''
    #''' code: 
    # Declaring values for b and net

    b = np.zeros(2)
    b2 = np.zeros(1)
    net = [[np.identity(2), b], [np.array([[1,1]]), b2], [np.array([[1]]), b2]]
    nlf = 1 # non linear function = step-function

    #running one time
    y_in = np.array([0.2,-0.7]) # choose input
    y_out = apply_net(net, y_in, nlf) # apply net

    #plotting
    Plot_2Dnetwork2(nlf) # plotting all possible values
    #'''
   
   
