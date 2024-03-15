# Supervised Learning
Supervised learning is a foundational concept in machine learning where algorithms learn from labeled data to make predictions or decisions. In this paradigm, each input data point is associated with a corresponding output label, allowing the algorithm to learn the mapping between input and output. Through iterative training on labeled examples, supervised learning models generalize patterns and relationships in the data, enabling accurate predictions on unseen data. This approach finds applications in classification, regression, and many real-world scenarios. Some examples are given below.
# Linear Regression

Linear regression models the relationship between dependent and independent variables using a straight line. It minimizes the sum of squared differences between observed and predicted values. Widely used in fields like economics and finance, it predicts outcomes and assesses variable impacts.
A important measurement for a linear regression algorithm is the R-squared (R2) score it is a statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, where 1 indicates a perfect fit, meaning the model explains all the variability in the response variable. 
A simple program to implement a basic linear regression algorithm is given below:
The libraries used are numpy,matplotlib,sklearn(scikit-learn).
```python
# import required modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Generate a random dataset
np.random.seed(42)
X = 2*np.random.randn(100,1)
y = 4+(3*X)+np.random.randn(100,1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
# Plot data for easy visualisation
plt.scatter(X_train,y_train,label = 'DataPlot')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.legend()
model = LinearRegression()# Create the model
model.fit(X_train,y_train) # Train the model
y_pred = model.predict(X_test) # Predict the model
mse = mean_squared_error(y_test,y_pred)
print(mse)
plt.scatter(X_test,y_test,color = 'black',label = 'Test')
plt.plot(X_test,y_pred, lw = 3, label = 'predicted')# Plottting the prediction for easy visualisation
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.legend()
r2 = r2_score(y_test, y_pred)#
print("R2 --> ", r2)
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/8fe6c3bb-336a-46ce-af82-05514e0818e1)
# Lasso (Least Absolute Shrinkage and Selection Operator) and Ridge regression are regularization techniques used in linear regression to prevent overfitting and improve the model's generalization performance.

- Lasso Regression:

- Regularization Term: L1 regularization.
-- Objective Function: Minimizes the sum of squared errors plus the absolute values of the coefficients multiplied by a regularization parameter (alpha).
-- Effect: Encourages sparsity in the model by driving some coefficients to exactly zero. It is useful for feature selection.
-Ridge Regression:

-- Regularization Term: L2 regularization.
-- Objective Function: Minimizes the sum of squared errors plus the squared magnitudes of the coefficients multiplied by a regularization parameter (alpha).
-- Effect: Reduces the magnitude of the coefficients, preventing them from becoming too large. It is effective when there is multicollinearity among the independent variables.

The programs for basic regularised models using these methods can be found here:![Creating-LinearRe-DecisionTree-Models](https://github.com/VMOnGit/Python-for-Basic-ML/tree/main/Creating-LinearRe-DecisionTree-Models)

# Logistic Regression

Logistic regression is a supervised learning algorithm used for binary classification. It models the probability of an instance belonging to a specific class, employing the logistic function to produce values between 0 and 1. A threshold is applied to predict the class, making it effective for categorical outcomes.
## Confusion Matrix:

A confusion matrix is a table that summarizes the performance of a classification algorithm. It compares predicted and actual values, categorizing them into true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

## Accuracy
Accuracy:
Accuracy measures the overall correctness of a classification model. It is calculated as the ratio of correct predictions (TP + TN) to the total number of predictions.

Accuracy = TP+TN+FP+FN/TP+TN

## Recall
Recall quantifies the ability of a model to identify all relevant instances of a class. It is calculated as the ratio of true positives (TP) to the sum of true positives and false negatives (FN).
A basic logistic regression implementation is given below:
## importing the requried libraries (sklearn,numpy,pandas)
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
```
## Reading, splitting, scaling data
```python
csv_file_path = 'Social_Network_Ads.csv'
df = pd.read_csv(csv_file_path)#Read the file
x = df.iloc[:, [2, 3]].values#Grab data
y = df.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)#Split train and test

sc = StandardScaler()#Scale down the data
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```
## Make the model, fit the model, predict the model
```python
#Make the classifier model
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
```
## Make confusion matrix, measure accuracy, precision, recall
```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test,y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['Negative', 'Positive']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45,visible = True)
plt.yticks(tick_marks, classes,visible = True)

plt.xlabel('Predicted label')
plt.ylabel('True label')

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center', color='white' if cm[i, j] > conf_matrix.max() / 2 else 'black')

plt.show()
```
![image](https://github.com/VMOnGit/Python-for-Basic-ML/assets/114856002/e6c1a33b-f46f-4633-9654-c5d4bb113be5)

These are 2 prime examples for supervised learning and more models based on decision tree and random forest can be found here:![Creating-LinearRe-DecisionTree-Models](https://github.com/VMOnGit/Python-for-Basic-ML/tree/main/Creating-LinearRe-DecisionTree-Models)
