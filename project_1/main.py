import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("UCI_Credit_Card.csv")

# NULL CHECKER
# for idx in range(25):
#     print("NULL VALUES in column", idx , "are")
#     null_cnt = df.iloc[:,idx].isnull().sum()
#     print(null_cnt)
#The dataset is excellent so preprocessing not required
    

default_payment = df['default.payment.next.month']

new_df = df.drop(['ID', 'default.payment.next.month'], axis = 'columns') #2 columns which isn't really required

#using for Normalization
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
new_df.iloc[ : , 12:24] = scale.fit_transform(new_df.iloc[ : , 12:24])

# Visualize the distribution of features using histograms
plt.figure(figsize=(15, 10))
for i, column in enumerate(new_df.columns, 1):
    plt.subplot(4, 6, i)
    sns.histplot(new_df[column], kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(new_df.corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

#We now split our dataset into TRAINING and TESTING SPLITS and 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(new_df, default_payment, train_size = 0.8, random_state = 6)

#logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_predicted = lr.predict(X_test)
print("The first five values predicted are: ")
print(Y_predicted[0:10])

print("The first five test values are: ")
print(np.array(Y_test[0:5]))

print("The accuracy of the LogisticRegression model is ")
print(lr.score(X_test, Y_test))

#generating confusion matrix
from sklearn import metrics
matrix_logistic = metrics.confusion_matrix(Y_predicted, Y_test)
sns.heatmap(matrix_logistic, annot = True, cmap = "Greens", fmt = '0.1f')
plt.xlabel("Predicted Values")
plt.ylabel("Test Values")
plt.title("Confusion Matrix for LogisticRegression")
plt.show()

#GaussianNB classification model
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_predicted = nb.predict(X_test)
print("The first five values predicted are: ")
print(Y_predicted[0:10])

print("The first five test values are: ")
print(np.array(Y_test[0:5]))

print("The accuracy of the NaiveBayesClassifier model is ")
print(nb.score(X_test, Y_test))

#cofusion matrix
matrix_naive = metrics.confusion_matrix(Y_predicted, Y_test)
sns.heatmap(matrix_naive, annot = True, cmap = "Blues", fmt = '0.1f')
plt.xlabel("Predicted Values")
plt.ylabel("Test Values")
plt.title("Confusion Matrix for NaiveBayes")
plt.show()

#CROSS VALIDATION SCORE --> helps in choosing the model
from sklearn.model_selection import cross_val_score
lr.fit(new_df, default_payment)
cross_val_score(lr, new_df, default_payment, cv = 10)
np.average(cross_val_score(lr, new_df, default_payment, cv = 10))


# Cross-validation score for logistic regression
cross_val_scores_lr = cross_val_score(lr, new_df, df['default.payment.next.month'], cv=10)

plt.figure(figsize=(8, 6))
sns.boxplot(data=cross_val_scores_lr, color='skyblue')
plt.title('Cross-Validation Scores for Logistic Regression')
plt.xlabel('Cross-Validation Fold')
plt.ylabel('Accuracy')
plt.show()