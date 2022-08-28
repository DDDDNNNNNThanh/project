#!/usr/bin/env python
# coding: utf-8

# Project: Titanic

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


train_df = pd.read_csv('C:\\Users\\Admin\\Desktop\\train.csv')
test_df = pd.read_csv('C:\\Users\\Admin\\Desktop\\test.csv')


# In[13]:


train_df.columns


# In[14]:


test_df.columns


# ![image.png](attachment:image.png)

# In[15]:


#preview data
train_df.head()


# In[16]:


train_df.set_index(train_df.PassengerId, inplace=True)


# In[17]:


train_df


# In[20]:


train_df.drop('PassengerId', axis = 1, inplace=True)


# In[21]:


train_df


# In[18]:


test_df = pd.read_csv('C:\\Users\\Admin\\Desktop\\test.csv', index_col='PassengerId')


# In[19]:


test_df.head()


# In[22]:


test_df.info()


# In[23]:


train_df.info()


# In[24]:


train_df['Survived'] = train_df['Survived'].astype('category')


# In[25]:


train_df['Survived'].dtype


# In[26]:


train_df.info()


# In[27]:


features = ['Pclass','Sex','SibSp','Parch','Embarked']
def convert_fearures(df,features):
    for feature in features:
        df[feature] = df[feature].astype('category')
convert_fearures(train_df,features)
convert_fearures(test_df,features)


# In[28]:


train_df.info()


# 1.1.1. Distribution of Numerical feature values across the samples

# In[29]:


train_df.describe()


# 1.1.2. Distribution of Categorical features

# In[30]:


train_df.describe(include=['category'])


# # Exploratory Data Analysis (EDA)

# # Target Variable = Survived

# In[31]:


train_df['Survived'].value_counts().to_frame()


# In[33]:


train_df['Survived'].value_counts(normalize=True).to_frame()


# Only 38% survived in the disaster. So the training data suffer from data imbalance but it is nor severe so i won't consider the techinique like sampling to tackle the imbalance
# 
# # Sex

# In[34]:


train_df['Sex'].value_counts().to_frame()


# In[35]:


train_df['Sex'].value_counts(normalize=True).to_frame()


# In[42]:


sns.countplot(data=train_df,x='Sex',hue='Survived',palette='Blues')


# # Remaining Categorical Feature Columns

# In[54]:


cols = ['Pclass','Sex','SibSp','Parch','Embarked']

n_rows = 2
n_cols = 3

fig , ax = plt.subplots(n_rows,n_cols,figsize=(n_cols*3.5, n_rows*3.5))

for r in range(0,n_rows):
    for c in range(0,n_cols):
        i = r*n_cols + c
        if i < len(cols):
            ax_i=ax[r,c]
            sns.countplot(data=train_df, x=cols[i], hue='Survived', palette='Blues',ax=ax_i)
            ax_i.set_title(f"Figure{i+1}: Survived rate vs {cols[i]}")
            ax_i.legend(title='',loc='upper right', labels=['Not survived','Survived'])
ax.flat[-1].set_visible(False) #Remove last subplot
plt.tight_layout()


# # OBSERVATIONS:
# # Survived:
#   - Fig1: Female survival rate > Male
#   - Fig2: Most people embarked from Sothampton and also had the highest survival rate
#   - Fig3: 1st class have higher survival rate
#   - Fig4: People going with 0 sibling were mostly not survived.But the number of people going with 1 or 2                           sibling had a better chance of living
#   - Fig5: People going with 0 Parch were almost not survived

# # Numerical Features

# # Age

# In[62]:


sns.histplot(data=train_df,x='Age',hue='Survived',bins=40,kde=True)


# - Major passengers were from 1840 ages
# - Children had more chance of surviving

# # Fare

# In[72]:


train_df['Fare'].describe()


# In[71]:


# To name for 0-25% quartitle, 25-50, 50-75,75-100
fare_categories = ['Economics','Standard','Expensive','Luxury']
quartitle_data = pd.qcut(train_df['Fare'],4,labels=fare_categories)
sns.countplot(x=quartitle_data,hue=train_df['Survived'])


# Distribution of Fare:
#   - fare doesn't follow a normal distribution and has a huge spike at the price range(0-100)
#   - The distribution is skewed to the left with 0.75 of the fare 31 and a max paid of 512
# Quartitle plot:
#   - Passengers in Luxury group have more chance to survive

# In[ ]:




