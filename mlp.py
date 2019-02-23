#!/usr/bin/env python
# coding: utf-8

# # MLP 0
# ---

# In[31]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_diff(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_diff(x):
    return 1.0 - x**2


# In[53]:


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        self.acti = activation
        
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_diff = sigmoid_diff
            
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_diff = tanh_diff

        # Set weights
        
        self.weights = []
        
        # layers = [2,2,1]
        # range of weight values (-1,1)
              
        # sono 3 anche le uscite perchè c'è un bias per ogni nodo di uscita, quindi aggiungo un valore in ingresso (1) 
        # e lo moltiplico per il bias nel prodotto np.dot(...) in mdo da sommare automaticamente il bias al prodotto!
        # in sostanza mettere un valore delle x in più (aggiungo una dimensione), ed uno delle h (nodi intermedi aggiungo 
        # una dimensione)
        
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        #  pesi primo livello
        # | b1 w1.1 w1.2 | | 1  |     
        # | b2 w2.1 w2.2 | | x1 |  =  | a1 a2 a3 |
        # | b3 w3.1 w3.2 | | x2 |      
        #
        
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)

        # output layer - random((2+1, 1)) : 3 x 1
        #  pesi secondo livello
        # | b4 w4.1 w4.2 | | a1 |
        #                  | a2 |  =  | result |
        #                  | a3 |
        #
        
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)
        print(self.weights)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        print(X)
         
        for k in range(epochs):
            if k % 10000 == 0: print('epochs:', k)
            
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_diff(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_diff(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)     
        # print(a)
        # a = np.concatenate((np.array([[1]]), np.array([x])), axis=1)
        # print(a)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


# In[54]:


nn = NeuralNetwork([2,2,1])

# pallini
X_train = np.array([[0.375, 0.375],
                      [0.375, 0.625],
                      [0.625, 0.375],
                      [0.625, 0.625],
                    
                      [0.5, 0.25],
                      [0.5, 0.75],
                      [0.25, 0.5],
                      [0.75, 0.5]])
# quadrati
X_test = np.array([[0.5, 0.375],
                   [0.5, 0.625],
                  [0.375, 0.5],
                  [0.625, 0.5],
                   
                  [0.25, 0.25],
                  [0.75, 0.25],
                  [0.25, 0.75],
                  [0.75, 0.75]])

y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])

y_test = np.array([1, 1, 1, 1, 0, 0, 0, 0])


# In[55]:


nn.fit(X_train, y_train, 0.2, 100000)
print(nn.weights)


# In[56]:


for x1, x2, color in zip([x[0] for x in X_train], [x[1] for x in X_train], ["red" if x == 1 else "green" for x in y_train]):
    plt.scatter(x1, x2, marker='o', c=color)

for x1, x2, color in zip([x[0] for x in X_test], [x[1] for x in X_test], ["red" if x == 1 else "green" for x in y_test]):
    plt.scatter(x1, x2, marker='s', c=color)

# solo per forzare il plot a stare fra 0 e 1
plt.scatter(0,0,marker=".", c='white')
plt.scatter(1,1,marker=".", c='white')

plt.show()


# In[57]:


for e in X_test:
    pre = nn.predict(e)
    print(pre)
    plt.scatter(e[0], e[1], marker='s', c="orange" if pre[0] >= (0.5 if nn.acti == 'sigmoid' else 0.0) else "lime")
    
for x1, x2, color in zip([x[0] for x in X_train], [x[1] for x in X_train], ["red" if x == 1 else "green" for x in y_train]):
    plt.scatter(x1, x2, marker='o', c=color)

#for x1, x2, color in zip([x[0] for x in X_test], [x[1] for x in X_test], ["red" if x == 1 else "green" for x in y_test]):
#    plt.scatter(x1, x2, marker='s', c=color)

    

# solo per forzare il plot a stare fra 0 e 1
plt.scatter(0,0,marker=".", c='white')
plt.scatter(1,1,marker=".", c='white')

plt.show()


# In[58]:


c = np.zeros((10,10))
print(len(c))

for i in range(len(c)):
    for j in range(len(c[i])):
        c[i][j] = nn.predict([i/10, j/10])

# print(c)


# In[30]:


for i in range(len(c)):
    for j in range(len(c[i])):
        plt.scatter(i/10, j/10, c="red" if c[i][j] > 0.5 else "green")

for x1, x2, color in zip([x[0] for x in X_train], [x[1] for x in X_train], ["red" if x == 1 else "green" for x in y_train]):
    plt.scatter(x1, x2, marker='o', c=color)

for x1, x2, color in zip([x[0] for x in X_test], [x[1] for x in X_test], ["red" if x == 1 else "green" for x in y_test]):
    plt.scatter(x1, x2, marker='s', c=color)

plt.show()


# In[ ]:





# In[ ]:




