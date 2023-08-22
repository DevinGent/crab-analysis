import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# From exploration.py we saw that the various weights had a roughly linear relationship.  We now perform regression on these factors.

df=pd.read_csv('CrabAgePrediction.csv')
df.info()
# We select only those columns directly related to weight.
df=df[['Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']]
df.info()
print(df)
print(df.corr())

train_df, test_df = train_test_split(df,train_size=.8, random_state=1)
# We split our dataframe into a training section (to build our models) and a test section (to test our models).

#########################################################################################################
# First let us use weight to predict shucked weight.
sns.scatterplot(data=df,x='Weight',y='Shucked Weight')
plt.show()

lin_fit=np.polynomial.Polynomial.fit(train_df['Weight'],train_df['Shucked Weight'],1).convert()
# Use convert to get a nice polynomial out (1 indicates degree 1: a linear function)
print(lin_fit)
# We can get the y value for a given x value by using
print(lin_fit(3))
# If we just want the coefficients we use
print(lin_fit.coef)
# Comparing the two methods of evaluating the function.
print(lin_fit.coef[0]+lin_fit.coef[1]*3)

plt.figure(figsize=(8,6))
sns.scatterplot(data=train_df, x='Weight', y='Shucked Weight', label='Actual (Training)')
sns.lineplot(x=train_df['Weight'],y=lin_fit(train_df['Weight']), color='C1', label='Predicted')
plt.title('Linear model')
plt.legend()
plt.show()
print('The Coefficient of Determination on the training data is', r2_score(train_df['Shucked Weight'],lin_fit(train_df['Weight'])))

# Now for how it works on the testing data.

plt.figure(figsize=(8,6))
sns.scatterplot(data=test_df, x='Weight', y='Shucked Weight', label='Actual (Testing)')
sns.lineplot(x=test_df['Weight'],y=lin_fit(test_df['Weight']), color='C1', label='Predicted')
plt.title('Linear model')
plt.legend()
plt.show()
print('The Coefficient of Determination on the testing data is', r2_score(test_df['Shucked Weight'],lin_fit(test_df['Weight'])))
# r2_score always uses (REAL y, PREDICTED y)

# We will also test how this works using scikit-learn.
X_test= np.array(test_df['Weight']).reshape(-1, 1)
X_train=np.array(train_df['Weight']).reshape(-1, 1)
y_test= np.array(test_df['Shucked Weight']).reshape(-1, 1)
y_train=np.array(train_df['Shucked Weight']).reshape(-1, 1)
linreg= LinearRegression()
linreg.fit(X_train, y_train)
print(linreg.score(X_test, y_test))
print(linreg)
print(linreg.coef_)
print(linreg.intercept_)

# Let use try using multiple regression now, using weight, shell weight, and viscera weight 
# to predict shucked weight and compare to the simple linear model
multireg = LinearRegression()
multireg.fit(train_df[['Weight', 'Viscera Weight', 'Shell Weight']].to_numpy(), train_df['Shucked Weight'])
# The .to_numpy() is only required to stop a warning later on (when predicting) about the regression being fitted with labels. 
# We could also just feed the predict a dataframe with the right labels.  I'll leave an example below.

print("The score for the multiple regression model is",
      multireg.score(test_df[['Weight', 'Viscera Weight', 'Shell Weight']], test_df['Shucked Weight']))

################################################################################
# Let us use these models to try and predict shell weight against a few examples from the test set.
print(test_df)
# Here are a few of the entries: 
# (the columns are index, weight, shucked weight, viscera weight, shell weight)
""""
1413  35.507749       16.584457        7.356695      9.355335
2785  30.518237       13.919605        6.236890      8.136306
2905  35.507749       20.695135       11.268926     13.097469
"""
# We will round these out and use the predict method of each linear model.  We will test the following:
"""
36      ??      7       9
31      ??      6       8
36      ??      11      13      
"""
# First let's use the linear model.
print("The linear model predicts that a crab with weight 36 has shucked weight",linreg.predict(np.array([36]).reshape(-1,1)))
print("The linear model predicts that a crab with weight 31 has shucked weight",linreg.predict(np.array([31]).reshape(-1,1)))
# Let's also try with 35.5
print("The linear model predicts that a crab with weight 31.5 has shucked weight",linreg.predict([[31.5]]))
# Notice that the double brackets works, and we need not reshape the single entry.
print("The numpy model predicts that a crab with weight 36 has shucked weight",lin_fit(36))
print("The numpy model predicts that a crab with weight 31 has shucked weight",lin_fit(31))
print("The numpy model predicts that a crab with weight 31.5 has shucked weight",lin_fit(31.5))

print("The multiple regression model predicts that a crab with")
print("weight {}, viscera weight {}, and shell weight {} has shucked weight {}.".format(36,7,9, multireg.predict([[36,7,9]])))
print("The multiple regression model predicts that a crab with")
print("weight {}, viscera weight {}, and shell weight {} has shucked weight {}.".format(31,6,8, multireg.predict([[31,6,8]])))
print("The multiple regression model predicts that a crab with")
print("weight {}, viscera weight {}, and shell weight {} has shucked weight {}.".format(36,11,13, multireg.predict([[36,11,13]])))

# Testing predictions when features have labels:
multireg.fit(train_df[['Weight', 'Viscera Weight', 'Shell Weight']], train_df['Shucked Weight'])
predictdf=pd.DataFrame([[36,7,9],[31,6,8],[36,11,13]])
print(predictdf)
predictdf.columns=["Weight", "Viscera Weight", "Shell Weight"]
print(predictdf)
print(multireg.predict(predictdf))
