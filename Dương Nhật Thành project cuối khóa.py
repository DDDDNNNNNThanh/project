#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Import dataset
df = pd.read_csv('C:\\Users\\Admin\\Desktop\\MCI(python)\\hmeq.csv')


# In[3]:


df.head()


# * **BAD**: 1 = client defaulted on loan 0 = loan repaid
# * **LOAN**: Amount of the loan request/ Giá trị khoản vay mong muốn giải ngân
# * **MORTDUE**: Amount due on existing mortgage/ Giá trị khoản vay mortgage hiện tại của khách hàng
# * **VALUE**: Value of current property/ Giá trị tài sản thế chấp
# * **REASON**: DebtCon = debt consolidation HomeImp = home improvement/ Lý do vay vốn
# * **JOB**: Six occupational categories/ Chức vụ công việc của khách hàng
# * **YOJ**: Years at present job/ Số năm làm việc tại công việc hiện tại
# * **DEROG**: Number of major derogatory reports/ Số lần được ghi nhận hành vi tín dụng xấu
# * **DELINQ**: Number of delinquent credit lines/ Số lượng các khoản vay quá hạn
# * **CLAGE**: Age of oldest trade line in months/ Thời gian từ lần phát sinh tín dụng đầu tiên tới hiện tại (theo tháng)
# * **NINQ**: Number of recent credit lines/ Số lượng khoản tín dụng gần đây
# * **CLNO**: Number of credit lines/ Tổng số lần phát sinh tín dụng
# * **DEBTINC**: Debt-to-income ratio/ Tỷ lệ Nơ/Thu nhập

# In[4]:


df.dtypes


# In[43]:


# Define columns in usage
label = ['BAD']
removed_features = ['LOAN','MORTDUE','VALUE','YOJ','CLAGE','NINQ','CLNO']
features = [col for col in list(df.columns) if col not in removed_features + label]

#### Identify numerical and categorical features
num_features = ['DEROG', 'DELINQ', 'DEBTINC']
cat_features = ['REASON', 'JOB']


# In[ ]:





# ### A. Train/Test Split

# In[44]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)


# In[45]:


# Train set
train_label = train_set[label]
train_cat = train_set[cat_features]
train_num = train_set[num_features]

# Test set
test_label = test_set[label]
test_cat = test_set[cat_features]
test_num = test_set[num_features]


# ### B. Base Model

# **Base model** is used to compare with other model versions.
# 
# We build base model using Logistic Regression.
# To determine base model, we need to ensure pre-requisites:
# * Missing values were treated
# * Categorical features were encoded to numerical values

# #### 1. Categorical features encoding

# In[46]:


cat_features


# In[47]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="constant", fill_value="Missing")
train_cat_array = imputer.fit_transform(train_set[cat_features])
train_cat_array


# In[48]:


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
train_cat_ordinal = ordinal_encoder.fit_transform(train_cat_array)
train_cat_array = train_cat_ordinal
train_cat_array


# In[49]:


encoded_cat_features = cat_features


# #### 2. Numerical Features treatment

# In[50]:


train_num.head()


# In[51]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="constant", fill_value=0)
train_num_array = imputer.fit_transform(train_num)


# In[52]:


train_num.head()


# #### 3. Form train set to fit model ####

# In[53]:


y_train_array = train_label.values
y_train_array.shape


# In[54]:


X_train_array = np.hstack((train_num_array, train_cat_array))
X_train_array.shape


# In[55]:


len(num_features + encoded_cat_features)


# #### 4. Form test set to validate model ####

# In[20]:


#Test label array
y_test_array = test_label.values
y_test_array.shape


# In[56]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="constant", fill_value="Missing")
test_cat_array = imputer.fit_transform(test_set[cat_features])
test_cat_array


# In[61]:


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
test_cat_ordinal = ordinal_encoder.fit_transform(test_cat_array)
test_cat_array = test_cat_ordinal
test_cat_array


# In[62]:


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
test_cat_ordinal = ordinal_encoder.fit_transform(test_cat_array)
test_cat_array = test_cat_ordinal
test_cat_array


# In[63]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="constant", fill_value=0)
test_num_array = imputer.fit_transform(test_num)


# In[64]:


y_test_array = test_label.values
y_test_array.shape


# In[65]:


X_test_array = np.hstack((test_num_array, test_cat_array))
X_test_array.shape


# #### 5. Fit model ####
# 
# <b>Logistic Regression</b>
# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# In[66]:


from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(C=1e8)
logit.fit(X_train_array, y_train_array)


# In[67]:


logit.coef_


# In[68]:


logit.intercept_


# In[69]:


logit.predict_proba(X_train_array)


# In[70]:


logit.classes_


# #### 6. Model validation with test set ####
# 
# In this project, your model needs to achieve AUC score above 0.7

# In[71]:


from sklearn.metrics import roc_auc_score, roc_curve, auc

test_scores = logit.decision_function(X_test_array)
test_auc_score = roc_auc_score(y_test_array, test_scores)
test_auc_score


# In[72]:


fpr, tpr, thresholds = roc_curve(y_test_array, test_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
plot_roc_curve(fpr, tpr)
plt.show()


# * **Other metrics**

# * **Confusion matrix**
# ![image.png](attachment:image.png)

# * **Precision vs Recall**

# ![image.png](attachment:image.png)
# _________________
# ![image-3.png](attachment:image-3.png)

# ### C. Model Improvement
# To impove model, we can consider the following points:
# * Get more data
# * Create new features with better prediction
# * Select right model
# * Feature scaling (or other feature transformations e.g. log,...)
# * Imputation (handle missing values)
# * Grouping/Binning/Label encoding
# * Remove multicolinearity (exclude features which are highly correlated with others)
# * Remove outliers
# * Regularization methods (e.g. Ridge, Lasso)
# * Tuning Hyper-parameters

# In[ ]:




