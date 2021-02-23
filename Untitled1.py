#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[4]:


pd.options.display.max_columns= None


# In[5]:


df = pd.read_csv('training_set_features.csv')
df


# In[6]:


tgt = pd.read_csv('training_set_labels.csv')
test = pd.read_csv('test_set_features.csv')
tgt.head(-2)
# test


# In[7]:


ndf = df.drop(columns=['respondent_id'])


# In[8]:


ntgt = tgt.drop(columns=['respondent_id'])


# In[9]:


big = pd.concat([ndf,ntgt],axis=1)


# In[10]:


big


# In[11]:


import seaborn as sns
sns.set(rc={'figure.figsize':(15.7,11.27)})


# In[12]:


big_corr = big.corr()
sns.heatmap(big_corr)


# In[13]:


big_h1 = big.drop(columns='seasonal_vaccine')


# In[14]:


big_h1


# In[15]:


big_h1.columns


# In[50]:


train,val = train_test_split(big_h1,test_size=0.2)
train.shape
val.shape


# In[20]:


big_h1.info()


# In[21]:


big_h1.isna().sum()


# In[23]:


from sklearn.impute import KNNImputer


# In[24]:


imputer = KNNImputer(n_neighbors=5)


# In[43]:


# a = imputer.fit_transform(big_h1[list(big_h1.iloc[:,:21].columns.values)])
b = imputer.fit_transform(big_h1[list(big_h1.iloc[:,31:33].columns.values)])


# In[44]:


big_h1.iloc[:,31:33] = b


# In[49]:


big_h1.isna().sum()


# In[48]:


# big_h1.fillna('Unknown',inplace=True)


# In[51]:


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('h1n1_vaccine')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


# In[52]:


b = 5
train_ds = df_to_dataset(train,shuffle=True,batch_size=b)
val_ds = df_to_dataset(val,batch_size=b)


# In[104]:


for f,l in train_ds.take(1):    #take 1 batch
      print('Every feature:', len(f.keys()))
      print('A batch of household_adults:', f['household_adults'])
      print('A batch of targets:', l )
    


# In[ ]:


next(iter(train_ds))[0]


# In[69]:


from tensorflow import feature_column
from tensorflow.keras import layers


# In[70]:


example_batch = next(iter(train_ds))[0]


# In[94]:


def demo(feat_col):
    f_layer = layers.DenseFeatures(feat_col)
    print(f_layer(example_batch).numpy())


# In[89]:


NUMERIC_COLS = list(big_h1.iloc[:,:21].columns)
NUMERIC_COLS.append('household_children')


# In[90]:


NUMERIC_COLS


# In[91]:


numeric_cols = [feature_column.numeric_column(x) for x in NUMERIC_COLS]


# In[95]:


for d in numeric_cols:
    demo(d)


# In[98]:


feat_cols


# In[121]:


vocab_cols = big_h1.select_dtypes(exclude=['float64','int64']).columns.values


# In[122]:


for x in list(vocab_cols):
    cat_col = feature_column.categorical_column_with_vocabulary_list(x, big_h1[x].unique())
    indicator_col = feature_column.indicator_column(cat_col)
    feat_cols.append(indicator_col)


# In[123]:


feat_cols


# In[125]:


feature_layer = tf.keras.layers.DenseFeatures(feat_cols)


# In[126]:


batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)


# In[127]:


model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dropout(.1),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[128]:


model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)


# In[1]:


test = pd.read_csv('test_set_features.csv')
test.head()


# In[ ]:




