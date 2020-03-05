# SVM (2D)

import os
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm, datasets

iris = datasets.load_iris()

# Take first two features
X = iris.data[:, :2]
y = iris.target

# SVM regularization parameter
C = 1.0 
svc = svm.SVC(kernel='linear', C=C, gamma='auto').fit(X, y)
# svc = svm.SVC(kernel='rbf', C=1, gamma=1).fit(X, y)

# Create a mesh (scope of every target value)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# Plot data points with predicted labels
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()
