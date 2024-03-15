# Unsupervised learning

Unsupervised learning is a machine learning approach where algorithms analyze unlabeled data to discover patterns, relationships, and structures without predefined outputs. Common tasks include clustering and dimensionality reduction. It is valuable for exploring data, revealing insights, and understanding inherent structures in diverse datasets.
A few types of unsupervised machine learning are:

## Hierarchial clustering
Hierarchical clustering is an unsupervised learning technique that organizes data into a tree-like structure, revealing relationships and similarities. It starts with individual data points as separate clusters and progressively merges them based on proximity, forming a hierarchy. The process continues until all points belong to a single cluster. This method provides insights into the inherent structure of the data, aiding in understanding complex relationships and organizing information hierarchically.
A program for a basic hierarchial clustering algorithm is given below:
```python
#loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.DataFrame([10,7,28,20,35],columns=["Marks"])
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(df, method='ward'))
plt.axhline(y=3, color='r', linestyle='--')
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/540cfb55-0cc6-4807-882e-a415e03d4612)

## K-Means clustering
K-Means clustering is an unsupervised machine learning algorithm used for partitioning a dataset into distinct groups or clusters. It aims to minimize the variance within each cluster and assigns data points to clusters based on their similarity to the cluster's centroid. The algorithm iteratively refines the cluster assignments until convergence, creating clusters that capture inherent patterns in the data. K-Means is widely applied in various domains, including customer segmentation, image compression, and anomaly detection.
Program for a basic K-means clustering algorithm is below:
```python
# Importing the packages

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
```
## Dataset read
```python
# Importing the dataset
iris=pd.read_csv("iris.csv")
sns.boxplot(x = 'species', y='sepal_length', data = iris)
```
## Dataset visualisation
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/4ab5d74e-2250-44d1-b2ed-47cc8e78c60a)
```python
sns.boxplot(x = 'species', y='sepal_width', data = iris)
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/fc607889-3660-429f-a32c-1cf0e0705f86)
```python
sns.boxplot(x = 'species', y='petal_length', data = iris)
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/b4ad8e98-9cc1-43f9-a479-e48e20ece385)
### Correlation plot
```python
figsize=[10,8]
plt.figure(figsize=figsize)
sns.heatmap(iris.drop(['species'],axis = 1).corr(),annot=True)
plt.show()
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/8cf77d30-5525-4221-bcc1-aadf696fb834)
## Finding ideal clusters(elbow method)
```python
ssw=[]
cluster_range=range(1,10)
for i in cluster_range:
    model=KMeans(n_clusters=i,init="k-means++",n_init=10, max_iter=300, random_state=0)
    model.fit(iris)
    ssw.append(model.inertia_)
ssw_df=pd.DataFrame({"no. of clusters":cluster_range,"SSW":ssw})
print(ssw_df)
plt.figure(figsize=(12,7))
plt.plot(cluster_range, ssw, marker = "o",color="cyan")
plt.xlabel("Number of clusters")
plt.ylabel("sum squared within")
plt.title("Elbow method to find optimal number of clusters")
plt.show()
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/0b1f1b16-2c61-483d-893b-f5c75261010b)
```python
## It returns the cluster vectors i.e. showing observations belonging which clusters 
clusters=k_model.labels_
clusters
```
## Visualisation of the clusters
```python
iris = pd.read_csv("iris.csv")
sns.boxplot(x = 'clusters', y='petal_width', data = iris)
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/630c2c5f-7632-4b79-988b-16fc0a14a9e3)
```python
sns.boxplot(x = 'clusters', y='petal_length', data = iris)
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/2e523d84-8488-454b-8aa5-4cb9f7cde002)
```python
sns.pairplot(iris, hue='clusters')
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/2b639b74-de9d-4665-b82f-308be0ce0360)


