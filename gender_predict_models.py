import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Following our work in gender_prediction.py, we will now use various different models and compare their accuracy in predicting 
# crab genders.  The early part of this code is a direct copy from gender_prediction.py.

np.random.seed(1)

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
train_df, test_df = train_test_split(df,train_size=.8)
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

logr = LogisticRegression()
logr.fit(X_train.values, y_train)

print("From the testing data, a crab with")
print(X_test.iloc[0])
print("has gender",y_test.iloc[0])
print("The logistic model would predict gender", logr.predict(X_test.iloc[0].values.reshape(1,-1)))
###############################################################################################
# We will use a model number to save typing.
i=0
###############################################################################################
predicted = []
predicted.append(('Logistic Regression', logr.predict(X_test.values)))
print("The",predicted[i][0],'model predicts')
print(predicted[i][1])

c_matrices=[]
c_matrices.append((predicted[i][0], metrics.confusion_matrix(y_test, predicted[i][1], labels=[0,1])))
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrices[i][1], display_labels = ['Male', 'Female'])
cm_display.plot()
plt.show() 

accuracy=[]
accuracy.append((predicted[i][0],metrics.accuracy_score(y_test, predicted[i][1])))
print(accuracy[i])



# Next model (KNC)
i=i+1
kneigh = KNeighborsClassifier()
kneigh.fit(X_train.values, y_train)
predicted.append(('KNeighbors Classifier', kneigh.predict(X_test.values)))
print("The",predicted[i][0],'model predicts')
print(predicted[i][1])


c_matrices.append((predicted[i][0], metrics.confusion_matrix(y_test, predicted[i][1], labels=[0,1])))
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrices[i][1], display_labels = ['Male', 'Female'])
cm_display.plot()
plt.show() 

accuracy.append((predicted[i][0],metrics.accuracy_score(y_test, predicted[i][1])))
print(accuracy[i])

# Next model (LDA)
i=i+1
lda = LinearDiscriminantAnalysis()
lda.fit(X_train.values, y_train)
predicted.append(('Linear Discriminant Analysis', lda.predict(X_test.values)))
print("The",predicted[i][0],'model predicts')
print(predicted[i][1])

c_matrices.append((predicted[i][0], metrics.confusion_matrix(y_test, predicted[i][1], labels=[0,1])))
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrices[i][1], display_labels = ['Male', 'Female'])
cm_display.plot()
plt.show() 

accuracy.append((predicted[i][0],metrics.accuracy_score(y_test, predicted[i][1])))
print(accuracy[i])


# Next model (Gaussian)
i=i+1
gnb = GaussianNB()
gnb.fit(X_train.values, y_train)
predicted.append(('Gaussian NB', gnb.predict(X_test.values)))
print("The",predicted[i][0],'model predicts')
print(predicted[i][1])


c_matrices.append((predicted[i][0], metrics.confusion_matrix(y_test, predicted[i][1], labels=[0,1])))
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrices[i][1], display_labels = ['Male', 'Female'])
cm_display.plot()
plt.show() 

accuracy.append((predicted[i][0],metrics.accuracy_score(y_test, predicted[i][1])))
print(accuracy[i])


# Next model (SVC)
i=i+1
svc = SVC()
svc.fit(X_train.values, y_train)
predicted.append(('SVC', svc.predict(X_test.values)))
print("The",predicted[i][0],'model predicts')
print(predicted[i][1])


c_matrices.append((predicted[i][0], metrics.confusion_matrix(y_test, predicted[i][1], labels=[0,1])))
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrices[i][1], display_labels = ['Male', 'Female'])
cm_display.plot()
plt.show() 

accuracy.append((predicted[i][0],metrics.accuracy_score(y_test, predicted[i][1])))
print(accuracy[i])

# Next model (Decision Tree)
i=i+1
dtree = tree.DecisionTreeClassifier()
dtree.fit(X_train.values, y_train)
predicted.append(('Decision Tree', dtree.predict(X_test.values)))
print("The",predicted[i][0],'model predicts')
print(predicted[i][1])


c_matrices.append((predicted[i][0], metrics.confusion_matrix(y_test, predicted[i][1], labels=[0,1])))
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrices[i][1], display_labels = ['Male', 'Female'])
cm_display.plot()
plt.show() 

accuracy.append((predicted[i][0],metrics.accuracy_score(y_test, predicted[i][1])))
print(accuracy[i])


# Next model (Random Forest)
i=i+1
forest = RandomForestClassifier()
forest.fit(X_train.values, y_train)
predicted.append(('Random Forest', forest.predict(X_test.values)))
print("The",predicted[i][0],'model predicts')
print(predicted[i][1])


c_matrices.append((predicted[i][0], metrics.confusion_matrix(y_test, predicted[i][1], labels=[0,1])))
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrices[i][1], display_labels = ['Male', 'Female'])
cm_display.plot()
plt.show() 

accuracy.append((predicted[i][0],metrics.accuracy_score(y_test, predicted[i][1])))
print(accuracy[i])

# Next model (XGBoost)
i=i+1
xgb = XGBClassifier()
xgb.fit(X_train.values, y_train)
predicted.append(('XGBoost', xgb.predict(X_test.values)))
print("The",predicted[i][0],'model predicts')
print(predicted[i][1])


c_matrices.append((predicted[i][0], metrics.confusion_matrix(y_test, predicted[i][1], labels=[0,1])))
# Here labels should include the elements we are inputting and their order should match the display labels below.
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrices[i][1], display_labels = ['Male', 'Female'])
cm_display.plot()
plt.show() 

accuracy.append((predicted[i][0],metrics.accuracy_score(y_test, predicted[i][1])))
print(accuracy[i])
print(xgb.score(X_test.values,y_test))

# How do these score compared to just blindly picking male each time?
print("The accuracy when just picking male each time is",metrics.accuracy_score(y_test,[0]*len(y_test)))




