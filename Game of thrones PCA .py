#!/usr/bin/env python
# coding: utf-8

# In[1]:


#future is the missing compatibility layer between Python 2 and Python 3. 
#It allows you to use a single, clean Python 3.x-compatible codebase to 
#support both Python 2 and Python 3 with minimal overhead.
from __future__ import absolute_import, division, print_function


# In[2]:


#encoding. word encodig
import codecs
#finds all pathnames matching a pattern, like regex
import glob
#log events for libraries
import logging
#concurrency
import multiprocessing
#dealing with operating system , like reading file
import os
#pretty print, human readable
import pprint
#regular expressions
import re


# In[3]:


#natural language toolkit
import nltk
# word 2 vec
import gensim.models.word2vec as w2v
#dimensionality reduction
import sklearn.manifold
#math
import numpy as np
#plotting
import matplotlib.pyplot as plt
#parse dataset
import pandas as pd
#visualization
import seaborn as sns


# In[4]:


get_ipython().run_line_magic('pylab', 'inline')


# In[5]:


#stopwords like the at a an, unnecesasry
#tokenization into sentences, punkt 
#http://www.nltk.org/
nltk.download("punkt")
nltk.download("stopwords")


# In[6]:


#get the book names, matching txt file
book_filenames= sorted(glob.glob("/Users/poojasen/Documents/Project/word_vectors_game_of_thrones-LIVE-master/data/*.txt"))


# In[7]:


#print books
print("Found books:")
book_filenames


# In[8]:


#step 1 process data

#initialize raw unicode convert to UTF8 , we'll add all text to this one big file in memory
corpus_raw = u""
#for each book, read it, open it un utf 8 format, 
#add it to the raw corpus
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()


# In[9]:


#tokenizastion! saved the trained model here. Pickle - file format that we can load as a byte stream.
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[10]:


#tokenize into sentences
raw_sentences = tokenizer.tokenize(corpus_raw)


# In[11]:


#convert into list of words
#remove unecessary characters, split into words, no hyhens 
#split into words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words


# In[12]:


#for each sentece, sentences where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# In[13]:


#print an example
print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))


# In[14]:


#count tokens, each one being a sentence
token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))


# In[15]:


#step 2 build our model, another one is Glove
#define hyperparameters
#once we have vectors there are 3 main tasks- distance, similarity, ranking
#VECTOR IS A TYPE OF TENSOR

# Dimensionality of the resulting word vectors.
#more dimensions means more complex and expensive to train but also accurate and more generalize
#more training them
num_features = 300


# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words. 
#rate 0 and 1e-5 
#how often to look at the same word
#the more frequent the word is the less we want to use it to 
#create vectors, cause its already part of our trained model 
downsampling = 1e-3

# Seed for the Random Number Generator, to make the results reproducible.
#deterministic - good for debugging
seed = 1


# In[16]:


#We have imported the word2vec model from the gensim library
thrones2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


# In[17]:



thrones2vec.build_vocab(sentences)


# In[18]:


print("Word2Vec vocabulary length:", len(thrones2vec.wv.vocab))


# In[19]:


#train model on sentneces
# thrones2vec.train(sentences)
thrones2vec.train(sentences, total_examples=thrones2vec.corpus_count, epochs=thrones2vec.epochs)


# In[20]:


#save model
if not os.path.exists("trained"):
    os.makedirs("trained")


# In[21]:


thrones2vec.save(os.path.join("trained", "thrones2vec.w2v"))


# In[22]:


#load model
thrones2vec = w2v.Word2Vec.load(os.path.join("trained", "thrones2vec.w2v"))


# In[23]:


#squash dimensionality to 2
#https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm


from sklearn.decomposition import PCA


# In[24]:


#put it all into a giant matrix
all_word_vectors_matrix = thrones2vec.wv.syn0


# In[25]:



#train PCA
pca = PCA(n_components=50)
pca_result = pca.fit_transform(all_word_vectors_matrix)


# In[27]:


#plot point in 2d space
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix[thrones2vec.wv.vocab[word].index])
            for word in thrones2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)


# In[28]:


points.head(10)


# In[29]:


#plot
sns.set_context("poster")


# In[32]:


points.plot.scatter("x", "y", s=10, figsize=(20, 12))


# In[33]:


def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


# In[36]:


plot_region(x_bounds=(0.2, 0.4), y_bounds=(-0.2, 0.0))


# In[37]:


plot_region(x_bounds=(-0.4,-0.2), y_bounds=(-0.2, 0.0))


# 

# In[38]:


x_bounds=(-0.4,-0.2)
y_bounds=(-0.2, 0.0)
# print(points[(x_bounds[0]<=points.x) & (points.x<=x_bounds[1])])
print(points[(x_bounds[0]<=points.x) & (points.x<=x_bounds[1]) & (y_bounds[0] <= points.y) & (points.y <= y_bounds[1])])


# In[39]:


thrones2vec.most_similar("Stark")


# In[40]:


thrones2vec.most_similar("Aerys")


# In[41]:


thrones2vec.most_similar("pepper")


# In[42]:


thrones2vec.most_similar("Winterfell")


# In[43]:


#distance, similarity, and ranking
def nearest_similarity_cosmul(start1, end1, end2):
    similarities = thrones2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


# In[ ]:





# In[46]:


nearest_similarity_cosmul("Stark", "Winterfell", "Riverrun")
nearest_similarity_cosmul("Jaime", "sword", "wine")
nearest_similarity_cosmul("Arya", "Nymeria", "pepper")


# In[ ]:





# In[ ]:




