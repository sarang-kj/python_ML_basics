# Problem Statement: A study of the segmentation of the Intel Certification course participants over satisfaction level.

## Introduction:
- Feedback analysis is integral to assessing and improving the satisfaction levels of a course. It provides valuable insights into the effectiveness of content, instructional methods, and overall learning experience. Positive feedback reinforces successful elements, while constructive criticism identifies areas for refinement. This analysis bridges the gap between expectations and reality, guiding educators in adapting teaching strategies and content delivery. By actively seeking and acting upon feedback, educational institutions demonstrate a commitment to continuous improvement, ensuring a more satisfying and effective learning environment.

# Methodology

**Exploratory Data Analysis (EDA)** is a critical phase in the data analysis process where the primary focus is on gaining insights, summarizing main characteristics, and identifying patterns and trends within the data. It involves visualizing and summarizing the key features of a dataset to understand its underlying structure before formal modeling or hypothesis testing.

## Importance of Exploratory Data Analysis:

1. **Data Understanding:** EDA helps analysts develop a deeper understanding of the dataset, its variables, and their relationships. This understanding is crucial for informed decision-making.

2. **Pattern Recognition:** EDA allows the identification of patterns, trends, and anomalies in the data. This insight is valuable for formulating hypotheses and guiding further analysis.

3. **Data Cleaning:** Through EDA, data quality issues such as missing values, outliers, or inconsistencies are often discovered. Addressing these issues is essential for accurate and reliable analyses.

4. **Feature Selection:** EDA aids in the selection of relevant features for modeling by highlighting variables that are most informative or influential in explaining the variability in the data.

5. **Assumption Checking:** Before applying complex statistical models, EDA helps assess the assumptions and conditions required for these models. This ensures the validity of subsequent analyses.

6. **Communication:** EDA often involves creating visualizations that make it easier to communicate findings to stakeholders, facilitating a better understanding of the data and its implications.

This is conducted by reading the csv data in the form of a `pandas` dataframe. Data frames are easily manipulatable using `Python` programs making the above processes much easier.

# Machine Learning Approaches for Classification Problems

In machine learning, classification is a type of supervised learning task where the goal is to predict the categorical class labels of new instances based on past observations. There are several approaches for tackling classification problems:

1. **Logistic Regression:**
   - Logistic regression models the probability that a given instance belongs to a particular class. It is well-suited for binary classification problems.
   - The model uses the logistic function to map a linear combination of features to a value between 0 and 1, representing the probability.

2. **Decision Trees:**
   - Decision trees recursively split the data into subsets based on the most significant features, creating a tree-like structure.
   - Each leaf node corresponds to a class label. Decision trees are interpretable and easy to visualize but can be prone to overfitting.

3. **Random Forest:**
   - Random Forest is an ensemble method that builds multiple decision trees and combines their predictions.
   - It helps reduce overfitting and improves accuracy by aggregating the results of individual trees.

4. **Support Vector Machines (SVM):**
   - SVMs aim to find a hyperplane that best separates different classes in the feature space.
   - They work well for both linear and non-linear classification problems using kernel functions.

5. **K-Nearest Neighbors (KNN):**
   - KNN classifies instances based on the majority class among their k-nearest neighbors in the feature space.
   - It is a non-parametric, instance-based learning algorithm.

6. **Naive Bayes:**
   - Naive Bayes is based on Bayes' theorem and assumes that features are conditionally independent given the class.
   - It is particularly effective for text classification and is computationally efficient.

7. **Neural Networks:**
   - Neural networks, especially deep learning models, have gained popularity for complex classification tasks.
   - They consist of layers of interconnected nodes (neurons) and can automatically learn hierarchical representations.

8. **Gradient Boosting:**
   - Gradient Boosting builds a series of weak learners (typically decision trees) sequentially, with each tree correcting the errors of the previous ones.
   - It is powerful and often yields high accuracy, but it can be computationally intensive.

9. **Ensemble Methods:**
   - Ensemble methods, such as bagging and boosting, combine the predictions of multiple models to improve overall performance and robustness.
   
The choice of the classification algorithm depends on factors like the nature of the data, the size of the dataset, interpretability requirements, and the desired balance between bias and variance. It's common to experiment with multiple algorithms to determine which one performs best for a specific classification problem. For our purposes it is best to classify the data into clusters based on the satisfaction levels of different students and for clear differentiation between different satisfaction levels.

# Process of Exploratory Data Analysis(EDA) using `pandas`
## The steps to perform EDA are:
- Data loading
```python
import pandas as pd         #Importing the pandas library for data processing
df_class=pd.read_csv("https://raw.githubusercontent.com/sijuswamy/Intel-Unnati-sessions/main/Feed_back_data.csv")      #Pulling the dataset from GitHub
df_class.head()                         #Viewing the 1st five rows of the data to verify the dataset
```
- Data cleaning
```python
df_class.isnull().sum()
df_class = df_class.drop(['Timestamp','Email ID','Please provide any additional comments, suggestions, or feedback you have regarding the session. Your insights are valuable and will help us enhance the overall learning experience.'],axis=1)   #These datasets provide no help additonal data for our goal
df_class.info()                          #Gives info about every feature of the dataset
df_class.columns = ["Name","Branch","Semester","Resourse Person","Content Quality","Effeciveness","Expertise","Relevance","Overall Organization"]   #Assign better column names                                                                            #for easier understanding
   
```
- Exploratory Data Analysis
```python
round(df_class["Resourse Person"].value_counts(normalize=True)*100,2
round(df_class["Name"].value_counts(normalize=True)*100,2)
```
- Data visualisation
- The data is visualies on a bar and pie plot.
```python
ax = plt.subplot(1,2,1)
ax = sns.countplot(x='Resourse Person', data=df_class)
#ax.bar_label(ax.containers[0])
plt.title("Faculty-wise distribution of data", fontsize=20,color = 'Brown',pad=20)
ax =plt.subplot(1,2,2)
ax=df_class['Resourse Person'].value_counts().plot.pie(explode=[0.1, 0.1,0.1,0.1,0.1],autopct='%1.2f%%',shadow=True);
ax.tick_params(axis='x', labelrotation=90)
ax.set_title(label = "Resourse Person", fontsize = 20,color='Brown',pad=20);
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/a5013c3a-20fb-4a45-8e69-763b1c004c69)

- Summary of the data:

  The response of each student is assessed with respect to Content Quality,Expertise,Relevance,Overall Organization,Branch,Content Quality
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
plt.show()
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/a7075471-a757-417f-af6c-a3f1876bdf79)
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Effectiveness'])
plt.show()
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/86118186-cd7d-46d2-b9f5-9623d8f76e1d)
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Expertise'])
plt.show()
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/4789df7c-40a0-4f29-8b41-1bf4cad8a993)
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['relevance'])
plt.show()
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/bb291345-a4fc-461d-a29e-326ca66e48ad)
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Overall Organisation'])
plt.show()
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/4bd4371a-daaf-4075-9124-4536a9ff3e0b)
```python
sns.boxplot(y=df_class['Branch'],x=df_class['branch'])
plt.show()
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/88dab3b7-347b-4584-8e47-4c938af899f0)
```python
sns.boxplot(y=df_class['Branch'],x=df_class['Content Quality'])
plt.show()
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/f7a0b3da-9fc3-45d1-906d-615075d51192)

## Since we have completed EDA we move on to *creating, training and predicting* a K-Means clustering model

# Creating a K-Means model using scikit-learn or `sklearn`
For a K-Means clustering model, we need to identify the ideal amount of centroids to have the best output characteristics. There are 2 methods to identify this:
- Shoulder method
   - The shoulder method is a quicker, older method to find the ideal number  of centroids for a K-Means clustering algorithm.
- GridSearch method
   - In most use cases GridSearch is the better method as it yields better output characteristics.

## The steps to perform Shoulder method are:
We need to find the distance between the points and the centroids at which each cluster is properly differentiated and identified. The following code cell shows the method to do this using `sklearn` library to make use of the `KMeans` model to make the required calculations for us.
```python
input_col=["Content Quality","Effeciveness","Expertise","Relevance","Overall Organization"]
X=df_class[input_col].values
from sklearn.cluster import KMeans
wcss = []

# Try different values of k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)# here inertia calculate sum of square distance in each cluster
```
We then plot the distance between points in a cluster against the number of clusters to find the point where an "elbow" shape is formed. This point will be the ideal number of clusters
```python
# Plot the within-cluster sum of squares for different values of k
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.show()
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/9939e0ba-a2a0-4197-93a5-439e3d1a2eaf)

## The steps to perform Grid Search are:
- Define a parameter grid with different values of k for KMeans clustering.
- Use Grid Search Cross Validation to find the best parameters for KMeans clustering, optimizing for performance metrics such as mean squared error.
```python
# Define the parameter grid
from sklearn.model_selection import GridSearchCV

param_grid = {'n_clusters': [2, 3, 4, 5, 6]}

# Create a KMeans object
kmeans = KMeans(n_init='auto',random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=5)

# Perform grid search
grid_search.fit(X)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)
```
This will reveal the optimal number of centroids and the highest accuracy score possible.

# Now we move on to implementation of the K-Means Clustering Model
- Implement the K-Means clustering
```python
# Perform k-means clustering
k = 3 # Number of clusters
kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
kmeans.fit(X) #Trains the model on 'X'
```
- Extract the labels and cluster centers
```python
# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add the cluster labels to the DataFrame
df_class['Cluster'] = labels
```
- Visualise the clusters
```python
# Visualize the clusters
plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis')
plt.scatter(centroids[:,1], centroids[:, 2], marker='X', s=200, c='red')
plt.xlabel(input_col[1])
plt.ylabel(input_col[2])
plt.title('K-means Clustering')
plt.show()
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/9b6454d6-8f69-4cee-a42e-cd8972da3732)

# Conclusion:
Through this analysis we can see that data analytics forms an essential part of improvement and tailoring future sessions to better meet participant expectations, ultimately enhancing the overall learning experience.Valuable insights into participant perceptions and preferences regarding session quality that allows us to better understand the perspective of the participants.
