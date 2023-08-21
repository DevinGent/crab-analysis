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