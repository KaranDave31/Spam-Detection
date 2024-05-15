#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import nltk


# In[ ]:


#nltk.download_shell()


# In[1]:


messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]


# In[2]:


print(len(messages))


# In[3]:


messages[77]


# In[4]:


messages[57]


# In[5]:


for mess_no, message in enumerate(messages[:10]):
    print(mess_no,message)
    print('\n')


# In[6]:


messages[0]


# In[7]:


import pandas as pd


# In[8]:


messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['label','message'])


# In[9]:


messages.head()


# In[10]:


messages.describe()


# In[11]:


messages.groupby('label').describe()


# In[12]:


messages['length'] = messages['message'].apply(len)


# In[13]:


messages.head()


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


messages['length'].plot.hist(bins=100)


# In[16]:


messages['length'].describe()


# In[17]:


messages[messages['length'] == 910]['message'].iloc[0]


# In[18]:


messages.hist(column='length',by='label',bins=100,figsize=(12,4))


# In[19]:


import string


# In[20]:


mess = 'sample message !!!!!!!!! notice : is this'


# In[21]:


noPunc = [ c for c in mess if c not in string.punctuation]
noPunc


# In[22]:


from nltk.corpus import stopwords


# In[23]:


stopwords.words('english')


# In[24]:


noPunc = ''.join(noPunc)
noPunc


# In[25]:


noPunc.split()


# In[26]:


clean_mess = [word for word in noPunc.split() if word.lower() not in stopwords.words('english')]


# In[27]:


clean_mess


# In[28]:


def text_process(mess):
    
    nopunc = [char for char in mess if char not in string.punctuation]
    
    nopunc = ''.join(nopunc)
    
    return[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[29]:


messages.head()


# In[30]:


messages['message'].head(5).apply(text_process)


# In[31]:


from sklearn.feature_extraction.text import CountVectorizer


# In[32]:


bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])


# In[33]:


print(len(bow_transformer.vocabulary_))


# In[34]:


mess4 = messages['message'][3]
print(mess4)


# In[35]:


bow4 = bow_transformer.transform([mess4])
print(bow4)


# In[36]:


print(bow4.shape)


# In[37]:


bow_transformer.get_feature_names_out()[4068]


# In[38]:


bow_transformer.get_feature_names_out()[9554]


# In[39]:


messages_bow = bow_transformer.transform(messages['message'])


# In[40]:


print('shape of sparse matrix ', messages_bow.shape)


# In[41]:


messages_bow.nnz


# In[42]:


sparsity = (100*messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity {}'.format(sparsity))


# In[43]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[44]:


tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[45]:


tfid4 = tfidf_transformer.transform(bow4)
print(tfid4)


# In[46]:


tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]


# In[47]:


messages_tfidf = tfidf_transformer.transform(messages_bow)


# In[48]:


from sklearn.naive_bayes import MultinomialNB


# In[49]:


spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])


# In[51]:


spam_detect_model.predict(tfid4)[0]


# In[52]:


all_pred = spam_detect_model.predict(messages_tfidf)


# In[53]:


all_pred


# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size=0.3)


# In[61]:


msg_train


# In[62]:


from sklearn.pipeline import Pipeline


# In[63]:


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier', MultinomialNB())
])


# In[64]:


pipeline.fit(msg_train,label_train)


# In[65]:


predictions = pipeline.predict(msg_test)


# In[66]:


from sklearn.metrics import classification_report


# In[67]:


print(classification_report(label_test,predictions))


# In[ ]:




