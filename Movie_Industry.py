#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing Libraries

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) #adjusts the configuration of the plots we will create

# Read in the data

df = pd.read_csv(r'C:\Users\trero\Downloads\archive\movies.csv')


# In[4]:


#Looking at the data

df.head()


# In[8]:


#Let's see if there is any missing data

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[9]:


#Data types for our columns

print(df.dtypes)


# In[10]:


#Create correct year column

df['yearcorrect'] = df['released'].astype(str).str[:4]


# In[11]:


df.head()


# In[12]:


df.sort_values(by=['gross'], inplace=False, ascending=False)
df.head()


# In[13]:


pd.set_option('display.max_rows', None)


# In[14]:


df.head()


# In[16]:


#Drop any duplicates

df['company'].drop_duplicates().sort_values(ascending=False)


# In[ ]:


#Budget high correlation
#Company high correlation


# In[19]:


#Scatter plot with budget vs gross

plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')

plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')

plt.show


# In[18]:


df.head()


# In[25]:


#Plot the Budget versus Gross using Seaborn

sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color":"red"}, line_kws={"color":"blue"})


# In[29]:


#Looking at correlation

df.corr(method='pearson')


# In[31]:


correlation_matrix = df.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')

plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

plt.show()


# In[32]:


#Look at company

df.head()


# In[33]:


df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized.head()
        
        


# In[36]:


correlation_matrix = df_numerized.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')

plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

plt.show()


# In[37]:


correlation_mat = df_numerized.corr()

corr_pairs = correlation_mat.unstack()

corr_pairs


# In[38]:


sorted_pairs = corr_pairs.sort_values()

sorted_pairs


# In[39]:


high_corr = sorted_pairs[(sorted_pairs) > 0.5]

high_corr


# In[ ]:


#Votes and Budget have the highest correlation to gross earnings
#Company has low correlation

