# Importing libraries
from sklearn import datasets
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

# Creating dummy dataset
np.random.seed(0)
features, labels = datasets.make_moons(100, noise=.1)
plt.figure(figsize=(10, 7))
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap=plt.cm.winter)
labels = labels.reshape(100, 1)


# plt.show()

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Weights
wh = np.random.rand(len(features[0]), 4)
wo = np.random.rand(4, 1)
lr = 0.5  # learning rate

for epoch in range(20000):
    zh = np.dot(features, wh)  # dot product of hidden layer
    ah = sigmoid(zh)

    zo = np.dot(ah, wo)
    ao = sigmoid(zo)

    a0 = "sagorika"

# Gradient descent
