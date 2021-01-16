import numpy as np

import matplotlib.pyplot as plt

import time

import random

def sigmod(z):
    return 1/(1 + np.exp(-z))


x1_train = []
x2_train = []
y = []


def cost_function(X, Y, theta):

    H_theta = sigmod(np.dot(X, theta))

    cost = -(np.dot(Y.T, np.log(H_theta)) + np.dot((1 - Y).T, np.log(1 - H_theta)))

    return np.sum(cost)/len(Y)


def process_origin_data():
    
    with open('data.txt', 'r') as f:
        while True:
            line = f.readline()
            line = line.strip()
            line = line.strip('\n')
            line = line.strip('\r')
            if not line:
                for i in range(len(y)):
                    if y[i]: 
                        ax.plot(x1_train[i], x2_train[i], 'g.')
                        pass
                    else:
                        ax.plot(x1_train[i], x2_train[i], 'r.')
                        pass
                break
            
            x1_train.append(float(line.split('\t')[0]))
            
            x2_train.append(float(line.split('\t')[1]))
            
            y.append(int(line.split('\t')[2]))



def data_init():
    
    train_data1 = np.array(x1_train).reshape(len(x1_train), 1)

    train_data2 = np.array(x2_train).reshape(len(x2_train), 1)

    X = np.c_[train_data1, train_data2]

    X = np.c_[np.ones((len(x1_train),1)), X]

    Y = np.array(y).reshape(len(y), 1)

    theta = np.array([1,1,1]).reshape(3,1)

    return X, Y, theta


def gradient_descent(X, Y, theta, iters):

    cost = []

    last_time = time.time()

    for i in range(iters):

        alpha = (10-(9/iters*i))/1000

        H_theta = sigmod(np.dot(X, theta))

        theta = theta - alpha * np.dot(X.T, H_theta - Y)/len(Y)

        if i>100:

            cost.append(cost_function(X, Y, theta))

    print(time.time() - last_time)

    ax.plot(X[:,1], (theta[1]*(X[:,1]) + theta[0])/(-theta[2]))

    ax1.plot(range(101, iters), cost)


def stoc_grad_desc(X, Y, theta, iters):

    cost = []

    last_time = time.time() 

    for i in range(iters):

        alpha = (10-(9/iters*i))/5000

        index = random.randint(0, 99)

        H_theta = sigmod(np.dot(X[index], theta))

        H_theta = np.around(H_theta)

        theta = theta - (alpha * (X[index].reshape(len(X[index]),1)) * (H_theta - Y[index]))

        if i>100:
            cost.append(cost_function(X, Y, theta))

    print(time.time() - last_time)

    ax.plot(X[:,1], (theta[1]*(X[:,1]) + theta[0])/(-theta[2]))

    ax1.plot(range(101, iters), cost)


np.set_printoptions(suppress=True)

fig1 = plt.figure()

fig2 = plt.figure()

ax = fig1.add_subplot(111)

ax1 = fig2.add_subplot(111)

process_origin_data()

X, Y ,theta = data_init()

gradient_descent(X, Y, theta, 10000)

plt.show()

