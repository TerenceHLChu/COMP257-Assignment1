# Student name: Terence Chu
# Student number: 301220117

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
import numpy as np
from sklearn.datasets import fetch_openml, make_swiss_roll
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

#-----------
# Question 1
#-----------

# Retrieve and load MNIST 784 dataset of 70,000 instances 
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Assign data to X and target to y
X, y = mnist.data, mnist.target

# Display the first 10 digits in the MNIST dataset
plt.figure(figsize=(35, 6))
plt.suptitle('First 10 digits of MNIST dataset (784 dimensions)', fontsize=24, y=0.8)

for i in range(9):
    digit_image = X[i].reshape(28, 28)
    plt.subplot(1, 10, i + 1) # 1 row of subplots, 10 columns (i.e., 10 digits), place each image at index i + 1 (subplot indices start at 1)
    plt.imshow(digit_image, cmap='gray')
    plt.xticks([]) # Remove tick marks
    plt.yticks([])
    
plt.show()

# Retrieve the first 2 principal components 
n_components = 2

# Create an instance of PCA
pca = PCA(n_components = n_components)

# Fit the training data
X_2d_pca = pca.fit_transform(X)

# pca.explained_variance_ratio_ is an array containing 2 elements (n_components=2) - each element represents one dimension
# The zeroth index is the amount of original variance explained by the first principal component (i.e., 784 dimensions reduced to one)
# The first index is the amount of original variance explained by the second principal component (i.e., 784 dimensions reduced to two)
explained_variance = pca.explained_variance_ratio_

print('Explained variance ratio - first principal component', explained_variance[0])
print('Explained variance ratio - second principal component', explained_variance[1])

# Likewise, X_2d_pca is an array containing 2 elements
# X_2d_pca[:, 0] holds the first principal component (70,000 points)
# X_2d_pca[:, 1] holds the second principal componenet (70,000 points)
X_princ_comp_1 = X_2d_pca[:, 0]
X_princ_comp_2 = X_2d_pca[:, 1]

# Plot the projections of the first and second principal components onto a 1D hyperplane
fig = plt.figure(figsize=(30, 3))
plt.title('Projection of the first principal component onto a 1D hyperplane', fontsize=24)
plt.scatter(X_princ_comp_1, np.zeros_like(X_princ_comp_1), s=0.01) # x coordinates correspond to each of the 70,000 points in X_princ_comp_1, np.zeros_like assigns zeroes to the corresponding y coordinates
plt.show()

fig = plt.figure(figsize=(30, 3))
plt.title('Projection of the second principal component onto a 1D hyperplane', fontsize=24)
plt.scatter(X_princ_comp_2, np.zeros_like(X_princ_comp_2), s=0.01)
plt.show()

# Use incremental PCA to reduce the dimensionality down to 154 dimensions
n_components = 154
batch_size = 200 # Process the PCA in batch sizes of 200

# Create an instance of incremental PCA
ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
X_154d_ipca = ipca.fit_transform(X)

# Display the first 10 digits in the MNIST dataset (non-compressed)
plt.figure(figsize=(35, 6))
plt.suptitle('First 10 digits of MNIST dataset (784 dimensions)', fontsize=24, y=0.8)

for i in range(9):
    digit_image = X[i].reshape(28, 28)
    plt.subplot(1, 10, i + 1)
    plt.imshow(digit_image, cmap='gray')
    plt.xticks([]) 
    plt.yticks([])
    
plt.show()

# Map the 154 dimensions back to 784 to visually observe it
X_map_154d_to_784d = ipca.inverse_transform(X_154d_ipca)

# Display the first 10 digits in the MNIST dataset (compressed)
plt.figure(figsize=(35, 6))
plt.suptitle('First 10 digits of MNIST dataset (154 dimensions)', fontsize=24, y=0.8)

for i in range(9):
    reduced_digit_image = X_map_154d_to_784d[i].reshape(28, 28)
    plt.subplot(1, 10, i + 1) 
    plt.imshow(reduced_digit_image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
plt.show()

#-----------
# Question 2
#-----------

# Generate Swiss roll dataset
X, y = make_swiss_roll(n_samples=3000, random_state=42)

# Plot Swiss roll dataset
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection="3d")
ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y)
ax.set_title('Swiss roll dataset visualization', fontsize=24)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z') 
ax.view_init(azim=-70, elev=10)
plt.show()

# Reduce dimensionality with linear kernel PCA 
linear_pca = KernelPCA(n_components=2, kernel='linear')
X_reduced_linear = linear_pca.fit_transform(X)

# Plot the result of linear kernel PCA
fig = plt.figure(figsize=(8, 6))
plt.scatter(X_reduced_linear[:, 0], X_reduced_linear[:, 1], c=y)
plt.title('Swiss roll reduced to 2 dimensions with linear kPCA', fontsize=24)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Reduce dimensionality with RBF kernel PCA 
rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
X_reduced_rbf = rbf_pca.fit_transform(X)

# Plot the result of RBF kernel PCA
fig = plt.figure(figsize=(8, 6))
plt.scatter(X_reduced_rbf[:, 0], X_reduced_rbf[:, 1], c=y)
plt.title('Swiss roll reduced to 2 dimensions with RBF kPCA', fontsize=24)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Reduce dimensionality with sigmoid kernel PCA 
sigmoid_pca = KernelPCA(n_components=2, kernel='sigmoid', gamma=1E-3) # 0.003
X_reduced_sigmoid = sigmoid_pca.fit_transform(X)

# Plot the result of sigmoid kernel PCA
fig = plt.figure(figsize=(8, 6))
plt.scatter(X_reduced_sigmoid[:, 0], X_reduced_sigmoid[:, 1], c=y)
plt.title('Swiss roll reduced to 2 dimensions with sigmoid kPCA', fontsize=24)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Build pipeline to perform KPCA dimensionality reduction followed by classification with logistics regression
clf = Pipeline([
        ('kpca', KernelPCA(n_components=2)),
        ('log_reg', LogisticRegression())
    ])


# Define parameters for GridSearchCV to try
param_grid = [{
        'kpca__gamma': [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5],
        'kpca__kernel': ['linear', 'rbf', 'sigmoid']
    }]

# Convert y to categorical data (i.e., 0 and 1) for classification by logistics regression
for i in range(len(y)):
    if y[i] >= y.mean():
        y[i] = 1
    else:
        y[i] = 0

# Carry out hyperparameter tuning
grid_search = GridSearchCV(clf, param_grid, cv=10, verbose=3)
grid_search.fit(X, y)

print('The best parameters are:', grid_search.best_params_)

# Reduce dimensionality with the best parameters identified by GridSearchCV
rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.3)
X_reduced_rbf = rbf_pca.fit_transform(X)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X_reduced_rbf[:, 0], X_reduced_rbf[:, 1], c=y)
plt.title('Swiss roll reduced to 2 dimensions with the best parameters (kernel="rbf", gamma=0.3)', fontsize=24)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()