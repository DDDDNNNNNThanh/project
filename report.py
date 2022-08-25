#!/usr/bin/env python
# coding: utf-8

# In[112]:


import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


#  What was the best month for sales? How much was earned that month?

# In[113]:


frames = []
all_length = []
for file in os.listdir(path):
    if file.endswith('.csv'):
        filepath = path + file
        df1 = pd.read_csv(filepath) 
        frames.append(df1)
        result = pd.concat(frames)
        length_1month = len(df1.index)
        all_length.append(length_1month)

df = result
df.to_csv('annualSale2019.csv', index=False)


# In[19]:


df['Month'] = df['Order Date'].str[0:2]
df = df.dropna(how='all')
df = df[df['Month'] != 'Or']
df.head()


# In[25]:


df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], downcast='integer')
df['Price Each'] = pd.to_numeric(df['Price Each'], downcast='float')
df['Sales'] = df['Quantity Ordered'] * df['Price Each']
df.head()


# In[26]:


moving_column = df.pop('Sales')
df.insert(4, 'Sales', moving_column)
df.head()


# In[110]:


df.groupby('Month').sum()['Sales'].plot(kind='bar')


#  What city has the best sales?

# In[35]:


df.head()


# In[40]:


address_to_city = lambda address:address.split(',')[1]
df['City'] = df['Purchase Address'].apply(address_to_city)
df.head()


# In[41]:


df.groupby('City').sum()['Sales']


# In[42]:


sales_value_city = df.groupby('City').sum()['Sales']
sales_value_city.max()


# In[43]:


cities = [city for city, sales in sales_value_city.items()]
plt.bar(x=cities, height=sales_value_city)
plt.xticks(cities, rotation=90, size=8)
plt.xlabel('Cities')
plt.ylabel('Sales in USD')
plt.show()


# What time should we display ads to maximize the likelihood of customer’s buying product?

# In[44]:


df.head()


# In[48]:


df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Hours'] = df['Order Date'].dt.hour
df


# In[53]:


sales_value_hours = df.groupby('Hours').count()['Sales']
hours = [hour for hour, sales in sales_value_hours.items()]
plt.plot(hours, sales_value_hours)
plt.grid()
plt.xticks(hours, rotation=90, size=8)
plt.xlabel('Hours')
plt.ylabel('Sales in USD')
plt.show()


# What products are most often sold together?

# In[54]:


df.head()


# In[58]:


df_dup = df[df['Order ID'].duplicated(keep=False)]
groupProduct = lambda product: ', '.join(product)
df_dup['All Products'] = df_dup.groupby('Order ID')['Product'].transform(groupProduct)
df_dup.head()


# In[61]:


df_dup = df_dup[['Order ID', 'All Products']].drop_duplicates()
df_dup.head()


# In[64]:


df_dup['All Products'].value_counts().head(10)


# Task 5

# In[65]:


all_products = df.groupby('Product').sum()['Quantity Ordered']
all_products


# In[67]:


products_ls = [product for product, quant in all_products.items()]
plt.bar(products_ls, all_products)
plt.xticks(products_ls, rotation=90, size=8)
plt.xlabel('Products')
plt.ylabel('Quantity')
plt.show()


# What product sold the most? Why do you think it sold the most?

# In[71]:


#price
prices = df.groupby('Product').mean()['Price Each']
x = products_ls
y1 = all_products
y2 = prices

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(x, y1, color='g')
ax2.plot(x, y2, 'b-')

ax1.set_xticklabels(products_ls, rotation=90, size=8)
ax1.set_xlabel('Products')
ax1.set_ylabel('Quantity Ordered', color='g')
ax2.set_ylabel('Price Each', color='b')

plt.show()


# In[ ]:




