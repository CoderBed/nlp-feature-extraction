#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[ ]:


#LABEL ENCODER METHOD

from sklearn.preprocessing import LabelEncoder

categories = ['teacher', 'nurse', 'doctor', 'police']
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(categories)
df = pd.DataFrame({'Meslek': categories}, {'Etiket': encoded_labels})
df.head()


# In[ ]:


#ONE HOT ENCODİNG METHOD

from sklearn.preprocessing import OneHotEncoder
categories = ['teacher', 'nurse', 'doctor', 'police']
data = pd.DataFrame({'Meslek': categories})
encoder = OneHotEncoder (sparse_output = False, dtype = int)
encoded_data = encoder.fit_transform(data)
encoded_df = pd.DataFrame(encoded_data, columns = categories)
encoded_df.head()


# In[ ]:


#COUNT VECTORİZİNG METHOD (TF-IDF)

import pandas as pd
from sklearn.feature_extraction.text import TfidVectorizer

document = ['Bu ilk belgedir', 'İkinci belge budur', 'Ve üçüncü belgemiz', 'İlk belge hangisidir']
data = pd.DataFrame(('Text': documents))
vectorizer = TfidVectorizer()
tfidf_vector = vectorizer.fit_transform(data['text'])
tfidf_vector = pd.DataFrame(tfidf_vector.toarray(), columns = vectorizer.get_feature_names_out)
tfidf_vector.head()


# In[ ]:


#COUNT VECTORİZİNG METHOD (TF-IDF)

import nltk
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidVectorizer, CountVectorizer
from sklearn.decomposition import NMF, PCA
from nltk.tokenize import word_tokenize
from sklearn.manifold import TSNE
import spacy

text = "Doğal dil işleme bilgisayar bilimi alt alanıdır", "Yapay zeka ve hesaplamalı bilimdir", "Bilgisayar ve insan dili kesişimidir"
tokens = word_tokenize(text)
print(len(text), tokens[:50])
tfidf_vec = tfidVectorizer()

x_tfidf = tfidf_vec.fit_transform([text])
print(tfidf_vec.get_feature_names_out()[:50])
print(x_tfidf.toarray()([0] [:50]))


# In[ ]:


#COUNT VECTORİZİNG METHOD (BAG OF WORDS)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

documents = ['Bu ilk belgedir', 'İkinci belge budur', 'Ve üçüncü belgemiz', 'İlk belge hangisidir']
data = pd.DataFrame(('Text': documents))
vectorizer = CountVectorizer()
bow_vectors = vectorizer.fit_transform(data['Text'])
bow_df = pd.DataFrame(bow_vectors.toarray(), columns = vectorizer.get_feature_names_out()[:50])
bow_df.head()


# In[ ]:


#WORD EMBEDDİNG METHOD (WORD2VEC-CBOW)

import pandas as pd
from gensim.models import Word2Vec

sentences = ['Ben', 'severim', 'elmaları', 'Ben', 'yerim', 'meyve', 'elmalar', 'lezzetlidir', 'meyveler', 'sağlar', 'vitamin']
cbow = Word2Vec(sentences,min_count=1, vector_size=300, sg=0)
vectors = cbow.wv
vector_df = pd.DataFrame(vectors.vectors, index = vector.index_to_key)
vector_df.head(5)


# In[ ]:


#WORD EMBEDDİNG METHOD (WORD2VEC-SKİP GRAM)

import pandas as pd
from gensim.models import Word2Vec

sentences = ['Ben', 'severim', 'elmaları', 'Ben', 'yerim', 'meyve', 'elmalar', 'lezzetlidir', 'meyveler', 'sağlar', 'vitamin']
skip_gram = Word2Vec(sentences,min_count=1, vector_size=300, sg=1)
vectors = skip_gram.wv
vector_df = pd.DataFrame(vectors.vectors, index = vector.index_to_key)
vector_df.head(5)


# In[ ]:


#N-GRAM

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

documents = ['Bu ilk belgedir', 'İkinci belge budur', 'Ve üçüncü belgemiz', 'İlk belge hangisidir']
data = pd.DataFrame(('Text': documents))
vectorizer = CountVectorizer()
ngram_vectors = vectorizer.fit_transform(data['Text'])
ngram_df = pd.DataFrame(bow_vectors.toarray(), columns = ngram.get_feature_names_out()[:50])
ngram_df.head()

