import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

np.random.seed(1)
# This will make the results of the tree process consistent across different iterations of the script.

# We aim to predict the gender of crabs using their other features.
# For simplicity we will only consider crabs from our dataset which are known to be male or female.
# Our first task is to drop all entries with Indeterminate sex.

df=pd.read_csv('CrabAgePrediction.csv')
print(df)
print(df['Sex'].value_counts())
df=df[df['Sex']!='I']
df.reset_index(drop=True, inplace=True)
df.info()
print(df)

# When using the decision tree we want numerical imputs, so we will replace sex using the following dictionary:
# {'M':0,'F':1}

df['Sex'].replace({'M':0,'F':1},inplace=True)
print(df)

# We will split our data into a training and a testing portion.
train_df, test_df = train_test_split(df,train_size=.8, random_state=3)
train_df.info()
test_df.info()

# We will separate the predictors from the value we want to have predicted ('Sex')
X_train=train_df.drop('Sex',axis=1)
X_test=test_df.drop('Sex',axis=1)

X_train.info()
y_train=train_df['Sex']
y_test=test_df['Sex']
y_train.info()

# What are the gender counts for test vs train?
print(y_train.value_counts())
print(y_test.value_counts())
# These seem characteristic of the split in the population.

# We will create a small tree which is easier to visualize (hence max_depth=3)
small_dtree = tree.DecisionTreeClassifier(max_depth=3)
small_dtree = small_dtree.fit(X_train.values, y_train)

# We receive various warnings later if we try to use X right away which can be resolved by using .values here.  
# Let's verify what those values look like.
print(X_train.head(5))
print(X_train.head(5).values)

# Displaying the tree
plt.figure(figsize=(14,7))
tree.plot_tree(small_dtree, feature_names=[col for col in X_train.columns], fontsize=8) 
plt.show()


# Let's use the tree to predict a value:
print("The first few values of the testing set are:")
print(X_test.head(5))
print(y_test.head(5))
print("Taking the first row we have:")
print(X_test.iloc[0])
print('And gender',y_test.iloc[0])

print("The predicted sex was {}, the actual sex was {}".format(small_dtree.predict([X_test.iloc[0]]),y_test.iloc[0]))

# Let us get a column of predictions for all the test values and then compare them against the actual sexes using a confusion matrix.
predicted = small_dtree.predict(X_test.values)
print(predicted)

c_matrix = metrics.confusion_matrix(y_test, predicted, labels=[0,1])
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ['Male', 'Female'])
cm_display.plot()
plt.show() 
# From the confusion matrix we see that nearly all of the predictions ended up as 'Male'.  
# We will consider some other decision trees and see if they are better at predicting gender.

# The metric we will use in deciding which model is best is Accuracy: the ratio of correct predictions. 
# Note that Accuracy does not care about HOW a prediction might fail.  In our case it wouldn't capture the fact that
# this tree predicts male far more often than female.

# Accuracy will be between 0 and 1, with 1 being complete accuracy.
sm_model_accuracy=metrics.accuracy_score(y_test, predicted)
print("The accuracy of this model on the test data is",sm_model_accuracy)

# Let us try using a larger decision tree where we do not cap the depth.  This time we will not display the tree.
big_dtree = tree.DecisionTreeClassifier()
big_dtree = big_dtree.fit(X_train.values, y_train)

predicted=big_dtree.predict(X_test.values)
big_model_accuracy=metrics.accuracy_score(y_test, predicted)
print("The accuracy of the second model on the test data is",big_model_accuracy)

c_matrix = metrics.confusion_matrix(y_test, predicted, labels=[0,1])
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ['Male', 'Female'])
cm_display.plot()
plt.show() 



# What if we used fewer features originally?  
# For instance, we know from exploration.py that weight and shucked weight are correlated.  
# Is the redundency of certain factors giving them more weight?  Let's investigate.

X_train.info()
X_train=X_train[['Length','Height','Weight','Age']]
X_train.info()
X_test=X_test[['Length','Height','Weight','Age']]

col4_dtree = tree.DecisionTreeClassifier()
col4_dtree = col4_dtree.fit(X_train.values, y_train)

predicted=col4_dtree.predict(X_test.values)
col4_model_accuracy=metrics.accuracy_score(y_test, predicted)
print("The accuracy of the third model on the test data is",col4_model_accuracy)

c_matrix = metrics.confusion_matrix(y_test, predicted, labels=[0,1])
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = ['Male', 'Female'])
cm_display.plot()
plt.show() 
