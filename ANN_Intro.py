from sklearn import datasets
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(0)
feature_set, labels = datasets.make_moons(100, noise=.1)
plt.figure(figsize=(10,7))
plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap=plt.cm.winter)
labels=labels.reshape(100,1)

#plt.show()

