import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('CrabAgePrediction.csv')
df.info()
# Note that the dataframe contains no null entries.  We will check for duplicate elements.
print("We look for duplicate rows.  The shape of the dataframe of duplicate elements is:")
print(df[df.duplicated()].shape)
print("There are no duplicate rows.")

# We will save the row Sex as a category type.
df['Sex']=df['Sex'].astype('category')
df.info()

# How many crabs of each gender are there?
print(df['Sex'].value_counts())

# What is the correlation between the quantitative features?
corr_matrix=df.corr(numeric_only=True)
print(corr_matrix)

# It might help to visualize this with a heatmap.
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True)
plt.tight_layout()
plt.show()

# It seems that the age has less correlation with the other factors than they do with each other.

# Continuing this line of inquiry, let us consider scatterplots of the numeric columns.

numeric_df= df.drop('Sex', axis=1)


axes= pd.plotting.scatter_matrix(numeric_df, figsize=(11,8))
for ax in axes.flatten():
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')
plt.tight_layout()
plt.show()

# What are some takeaways from this?  Let's start by broadly categorizing the columns of our dataframe
# based on their scatterplots.
# Length and diameter seem to have a linear relation (so for our purposes will be somewhat interchangeable)
# Similarly weight, shucked weight, viscera weight, and shell weight have quasi-linear relationships with each other.
# Let us select a representative of each category so we are examining fewer columns at once.

slim_df=df[['Sex', 'Length','Height','Weight','Age']]

corr_matrix=slim_df.corr(numeric_only=True)
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True)
plt.tight_layout()
plt.show()


pd.plotting.scatter_matrix(slim_df.drop('Sex',axis=1), figsize=(8,6))
plt.tight_layout()
plt.show()


# Let's see if the correlation is impacted by gender.
male_corr=slim_df[slim_df['Sex']=='M'].corr(numeric_only=True)
fem_corr=slim_df[slim_df['Sex']=='F'].corr(numeric_only=True)
ind_corr=slim_df[slim_df['Sex']=='I'].corr(numeric_only=True)
vmin = min(male_corr.values.min(), fem_corr.values.min(),ind_corr.values.min())
vmax = max(male_corr.values.max(), fem_corr.values.max(),ind_corr.values.max())

fig, axs = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(10,6))

sns.heatmap(male_corr, annot=True,cbar=False,vmin=vmin,vmax=vmax, ax=axs[0])

sns.heatmap(fem_corr, annot=True,cbar=False,vmin=vmin,vmax=vmax,ax=axs[1])

sns.heatmap(ind_corr, annot=True,vmin=vmin,vmax=vmax,ax=axs[2])
axs[0].set_title('Male')
axs[1].set_title('Female')
axs[2].set_title('Indeterminate')
plt.tight_layout()
plt.show()