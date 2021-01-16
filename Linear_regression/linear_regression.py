import numpy as np

from matplotlib import pyplot as plt

from numpy import dot

#代价函数
def cost_function(X, Y, theta):
    
    cost = np.power((np.dot(X, theta) - Y), 2)

    return np.sum(cost)/(2*len(Y))

#梯度下降
def gradient_descent(X, Y, theta, alpha, iters):

    cost = []

    for i in range(iters):

        theta = theta - alpha * np.dot(X.T, np.dot(X, theta) - Y)/len(Y)

        cost.append(cost_function(X, Y, theta))

    print('============================')

    ax2 = fig1.add_subplot(212)

    ax2.plot(range(iters), cost)

    print(type(X))

    print(type(X[:,1]))

    print(X[:,1])

    ax1.plot(X[:,1], theta[1]*X[:,1]+theta[0], c='red')

    plt.show()

    return theta, cost[-1]


Z = np.ones((26,1))
 
X = np.array([0.067732,0.427810,0.995731,0.738336,0.981083,0.526171,0.378887,0.033859,0.132791,0.138306,0.247809,0.648270,0.731209,0.236833,0.969788,0.607492,0.358622,0.147846,0.637820,0.230372,0.070237,0.067154,0.925577,0.717733,0.015371,0.335070]).reshape(26,1)

X = np.column_stack((Z,X))

Y = np.array([3.176513,3.816464,4.550095,4.256571,4.560815,3.929515,3.526170,3.156393,3.110301,3.149813,3.476346,4.119688,4.282233,3.486582,4.655492,3.965162,3.514900,3.125947,4.094115,3.476039,3.210610,3.190612,4.631504,4.295890,3.085028,3.448080]).reshape(26,1)

fig1 = plt.figure()

ax1 = fig1.add_subplot(211)

ax1.plot(X[:,1], Y, 'b.')

theta = np.array([1,1]).reshape(2,1)

alpha = 0.1

iters = 100

theta, cost = gradient_descent(X, Y, theta, alpha, iters)

'''
x = np.array([1,2,3]).reshape(3,1)

y = np.array([2,4,6]).reshape(3,1)

theta = np.array([1.0]).reshape(1,1)

alptha = 0.2

print()

for i in range(10):
    
    print(np.sum((y-dot(x, theta))*x))
    
    theta = theta + alptha*np.sum((y-dot(x, theta))*x)/3.0
    
    print('x*theta = ' + str(dot(x, theta)))
    
    print('theta' + str(theta))
'''
