# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:40:43 2017

@author: Lenovo
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


data=pd.read_csv('finalizedfull.csv')
tweet=data['tweet']
senti_label=data['senti']
from sklearn.feature_extraction.text import CountVectorizer
c_vec=CountVectorizer()
bag_of_words=c_vec.fit_transform(tweet)
bag_dense=bag_of_words.todense()

x_train,x_test,y_train,y_test=train_test_split(bag_dense,senti_label,test_size=0.33)



import numpy as np
from sklearn.decomposition import PCA

pca=PCA(n_components=2)
pca.fit(x_train)        #these are numpy arrays
new_data=pca.transform(x_train)

x=[]
y=[]



for i in new_data:
    x.append(i[0])
    y.append(i[1])
    
print(x)
plt.scatter(x,y,c=y_train)
