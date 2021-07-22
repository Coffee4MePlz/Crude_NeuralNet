#Lesson1
import numpy as np
import matplotlib.pyplot as plt # for plotting
import matplotlib
import itertools

def apply_layer (y_in = [], w = [], b = []):
    '''
    y_in = previous layer results, i.e., a vector 1xN0
    b = next layer bias, i.e., a vector 1xN
    w = the connection weights, i.e., a NxN0 matrix 
    '''
    z = np.dot(w,y_in) +b
    y_out = sigmoid(z)

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
        if j+1 < len(Ns): 
            w = np.random.uniform(low=-10,high=10,size=(Ns[j+1],Ns[j]))
            b = np.random.uniform(low=-1,high=1,size=(Ns[j+1]))
            net.append([w,b])
        else: 
            b = np.random.uniform(low=-1,high=1,size=(Ns[j]))
            net.append([0,b])
    return net

def apply_net (net = [] ,y_in =[]):
    ''' net must be a list of arrays, each item j being 
    [w,b] = [weights matrix, bias vector]
    Function: Loops the layers
    '''
    for j in range(len(net)):
        if j+1 < len(net):
            y_in = apply_layer(y_in,net[j][0],net[j][1] )
        else: y_in = y_in + net[j][1]
    return y_in

def Plot_2Dnetwork(y_in = []):
    global net
    M = 100
    y_plot = np.zeros([M,M])
    for j1,j2 in itertools.product(range(M),range(M)):
        x = float(j1)/M -0.5
        y = float(j2)/M -0.5
        y_plot[j1,j2]=apply_net(net,[x,y])
    plt.imshow(y_plot,origin='lower',extent=(-0.5,0.5,-0.5,0.5))
    plt.colorbar()
    plt.title("NN output as a function of input values")
    plt.xlabel("y_2")
    plt.ylabel("y_1")
    plt.show()
    return 0

if __name__ == "__main__":
    Ns = [2,170,2100,40,500,20,5,1] #nodes per layer
    net = create_RandomNet(Ns) #calls for a random net
    y_in = np.array([0.2,-0.7])

    Plot_2Dnetwork(y_in)
    #apply_net(net,y_in)
