import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

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
train_df, test_df = train_test_split(df,train_size=.8, random_state=3)
train_df.info()
test_df.info()

X_train=train_df.drop('Sex',axis=1)
X_test=test_df.drop('Sex',axis=1)

X_train.info()
y_train=train_df['Sex']
y_test=test_df['Sex']
y_train.info()

small_dtree = tree.DecisionTreeClassifier(max_depth=3)
small_dtree = small_dtree.fit(X_train.values, y_train)
plt.figure(figsize=(14,7))
tree.plot_tree(small_dtree, feature_names=[col for col in X_train.columns], fontsize=8) 
plt.show()

# Let's use the tree to predict a value:
print(X_test.head(5))
print(y_test.head(5))
print("Taking the first row we have:")
print(X_test.iloc[0])
print('And gender',y_test.iloc[0])

print(small_dtree.predict([X_test.iloc[0]]))

predicted = small_dtree.predict(X_test.values)
print(predicted)

c_matrix = metrics.confusion_matrix(y_test, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = [False, True])

cm_display.plot()
plt.show() 