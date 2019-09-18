#!/usr/bin/env python
# coding: utf-8

# # Extract features of name from Training Text files

# In[128]:


import glob
import io
import os
import pdb
import sys

import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk


from nltk.chunk import tree2conlltags
import re
import pandas as pd


abc = []
persons = []
nwordsp= []
totchp = []
firstword = []
secondword = []
thirdword = []
fourthword = []
def get_entity(text):
    """Prints the entity inside of the text."""
    
    BC = u'\u2588'  
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                #print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
                #print (len(chunk))
                nwordsp.append(len(chunk))
                s=""
                k=0
                for c in chunk.leaves():
                    #print(c[0])
                    s+=c[0]
                    s+=" "
                    #print(s)
                    if len(chunk) == 1:
                        firstword.append(len(c[0]))
                        secondword.append(0)
                        thirdword.append(0)
                        fourthword.append(0)
                    if len(chunk) == 2 and k==0:
                        firstword.append(len(c[0]))
                    if len(chunk) == 2 and k==1:
                        secondword.append(len(c[0]))
                        thirdword.append(0)
                        fourthword.append(0)
                    if len(chunk) == 3 and k==0:
                        firstword.append(len(c[0]))
                    if len(chunk) == 3 and k==1:
                        secondword.append(len(c[0]))
                    if len(chunk) == 3 and k==2:
                        thirdword.append(len(c[0]))
                        fourthword.append(0)
                    if len(chunk) == 4 and k==0:
                        firstword.append(len(c[0]))
                    if len(chunk) == 4 and k==1:
                        secondword.append(len(c[0]))
                    if len(chunk) == 4 and k==2:
                        thirdword.append(len(c[0]))
                    if len(chunk) == 4 and k==3:
                        fourthword.append(len(c[0]))
                    k=k+1
                    
                    abc.append(c[0])
                persons.append(s[:-1])
                #print(len(s[:-1]))
                totchp.append(len(s[:-1]))  
#    print (persons)
#    print(len(persons))
#    for i in abc:
#        if i in text:
#            l=len(i)
#            text = text.replace(i,BC*l) 
#    print (text)
#    print (nwordsp)
#    print (totchp)
#    print (abc)
#    print (firstword)
#    print(len(firstword))
#    print (secondword)
#    print (thirdword)

    
def doextraction(glob_text):
    """Get all the files from the given glob and pass them to the extractor."""
    for thefile in glob.glob(glob_text):
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            get_entity(text)


 
#if __name__ == '__main__':
#    # Usage: python3 entity-extractor.py 'train/pos/*.txt'
doextraction('train/*.txt')


# # Getting the extracted features into a dataframe

# In[129]:


ytrain=pd.DataFrame(persons, columns=['Persons'])
xtrain = pd.DataFrame(nwordsp, columns=['count'])
xtrain['totalch']= totchp
xtrain['first']= firstword
xtrain['sec']= secondword
xtrain['third']= thirdword
xtrain['fourth']= fourthword

print(xtrain)
print (ytrain)


# # Using Machine learning metho Naive Bayes to train the model on train dataframe

# In[130]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
Xtrain=xtrain.iloc[:,0:5]
Ytrain=ytrain.iloc[:,0]
model1 = clf.fit(Xtrain, Ytrain)


# # Using  K nearest neighbors to train model

# In[131]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
model2 = knn.fit(Xtrain, Ytrain)


# # Redacting Files and saving them to Test 2 folder

# In[134]:


import glob
import io
import os
import pdb
import sys

import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk


from nltk.chunk import tree2conlltags
import re
import pandas as pd


abc = []
persons = []
nwordsp= []
totchp = []
firstword = []
secondword = []
thirdword = []
fourthword = []
count = []
firstw = []
secondw = []
thirdw = []
fourthw = []
totch = []
fileno = []
def get_entity(text):
    """Prints the entity inside of the text."""
    
    #BC = u'\u2588'
    BC = 'Q'
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                #print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
                #print (len(chunk))
                nwordsp.append(len(chunk))
                k=0
                for c in chunk.leaves():
                    abc.append(c[0])
                #persons.append(s[:-1])
                #print(len(s[:-1]))
                #totchp.append(len(s[:-1]))  
    
    for i in abc:
        if i in text:
            l=len(i)
            text = text.replace(i,BC*l) 
#    print (text)
    

    
    
    fileno.append(1)    
    filename = "test2/file"+ str(len(fileno))+".txt"
    file = open(filename, "w",encoding="utf-8")
    file.write(text) 
    file.close()

    #    print (nwordsp)
#    print (totchp)
#    print (abc)
#    print (firstword)
#    print(len(firstword))
#    print (secondword)
#    print (thirdword)

    
def doextraction(glob_text):
    """Get all the files from the given glob and pass them to the extractor."""
    
    for thefile in glob.glob(glob_text):
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            get_entity(text)
            


 
#if __name__ == '__main__':
#    # Usage: python3 entity-extractor.py 'train/pos/*.txt'
doextraction('test/*.txt')


# # Extracting features of redacted names from the redacted files

# In[136]:


import glob
import io
import os
import pdb
import sys

import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk


from nltk.chunk import tree2conlltags
import re
import pandas as pd


abc = []
persons = []
nwordsp= []
totchp = []
firstword = []
secondword = []
thirdword = []
fourthword = []
count = []
firstw = []
secondw = []
thirdw = []
fourthw = []
totch = []
fileno = []
def get_entity(text):
    """Prints the entity inside of the text."""
    
    #BC = u'\u2588'
    BC = 'Q'
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                #print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
                #print (len(chunk))
                nwordsp.append(len(chunk))
                k=0
                for c in chunk.leaves():
                    abc.append(c[0])
                #persons.append(s[:-1])
                #print(len(s[:-1]))
                #totchp.append(len(s[:-1]))  
    
    for i in abc:
        if i in text:
            l=len(i)
            text = text.replace(i,BC*l) 
#    print (text)
    
#    for i in nltk.word_tokenize(text):
#        m2 = re.search('^[\w]$', i)
#        if m2:
#            print("m")            
#        else:
#            print (i)
    
    array = []
    #count = []
    count1 = 0
    wc = 0
    wci = 0
    first = 0
    sec = 0
    third = 0
    fourth = 0
    count2 = 0
    for i in nltk.word_tokenize(text):
        
        m2 = re.search('^Q*Q$', i)
        if m2:
#            print(i)
            wci = wci +1
        else:
            wc = wc + 1
        if wci > wc:
            count1 = count1 + 1
        #    count2 = count2 + 1
        if count1 != 0 and wc > wci:
            count.append(count1)
            count1 = 0
        #if count1 = 1:
        #    first = len(i)
        wci = wc
    #if count2 == 0:
    #    ab = 0
    #    count.append(ab)
    print (count)
    count3 = 0
    wc = 0
    wci = 0
    tokens = nltk.word_tokenize(text)
#    print(tokens)
    j = 0
    for i in range(len(tokens)):
        m2 = re.search('^Q*Q$', tokens[i])
        if m2:
#            print(tokens[i])
            wci = wci +1
        else:
            wc = wc + 1
        if wci > wc:
            count3 = count3 + 1
#            print (count3)
        if count3 == 1:
            first = len(tokens[i])
            tot = first
#            print('first')
#            print(first)
        if count3 == 2:
            sec = len(tokens[i])
            tot = first + sec +1
#            print('second')
#            print(sec)
        if count3 == 3:
            third = len(tokens[i])
            tot = first + sec + third + 2
#            print(third)
        if count3 == 4:
            fourth = len(tokens[i])
            tot = first + sec + third + fourth + 3
#            print(fourth)
        if count3 == count[j]:
            count3 = 0
            firstw.append(first)
            secondw.append(sec)
            thirdw.append(third)
            fourthw.append(fourth)
            totch.append(tot)
            first = 0
            sec = 0
            third = 0
            fourth = 0
            
            
            if j < len(count)-1:
                j = j+1
        wci = wc
        tot =0
        
    #print("wc")
    print (firstw)
    print (secondw)
    print(thirdw)
    print(fourthw)
    print(totch)
#    for i in range (len(count)):
#        if count[i] == 1:
#            tot = firstw[i]
#        if count[i] == 2:
#            tot = firstw[i] + secondw[i] + 1
#        if count[i] == 3:
#            tot = firstw[i] + secondw[i] + thirdw[i] + 2
#       if count[i] == 4:
#            tot = firstw[i] + secondw[i] + thirdw[i] + fourth[i] + 3
#        totch.append(tot)
    

    #    print (nwordsp)
#    print (totchp)
#    print (abc)
#    print (firstword)
#    print(len(firstword))
#    print (secondword)
#    print (thirdword)

    
def doextraction(glob_text):
    """Get all the files from the given glob and pass them to the extractor."""
    
    for thefile in glob.glob(glob_text):
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            get_entity(text)
            


 
#if __name__ == '__main__':
#    # Usage: python3 entity-extractor.py 'train/pos/*.txt'
doextraction('test2/*.txt')


# # Getting the extracted test features into a dataframe

# In[137]:


#ytest=pd.DataFrame(persons, columns=['Persons'])
xtest = pd.DataFrame(count, columns=['count'])
xtest['totalch']= totch
xtest['first']= firstw
xtest['sec']= secondw
xtest['third']= thirdw
xtest['fourth']= fourthw

print(xtest)
#print (ytest)


# # Applying Naive Bayes and predicting the redacted names

# In[138]:


Xtest=xtest.iloc[:,0:5]
Ytest=ytest.iloc[:,0]


result = model1.predict(Xtest)
print(result)
#from sklearn.metrics import accuracy_score
#accuracy_score(result, Ytest)


# # Applying Kmeans and predicting the redacted names

# In[139]:


ypred = model2.predict(Xtest)
#print (knn.kneighbors(Xtest)[1])
print(ypred)
#from sklearn.metrics import accuracy_score
#accuracy_score(ypred, Ytest)


# # Getting top 5 names using kmeans clustering

# In[140]:


nearest5 = []
list
for i in range (len(knn.kneighbors(Xtest)[1])):
    top5 = []
    for j in range (len(knn.kneighbors(Xtest)[1][i])):
        m= knn.kneighbors(Xtest)[1][i][j]
        top5.append(Ytrain[m])
    nearest5.append(top5)
print(nearest5)
        


# # Calculating euclidean and cosine distances between names and getting top 5 names closest to the test name

# In[141]:


from scipy.spatial import distance
import numpy as np


distall = []
for i in range (len(Xtest)):
    distm = []
    cos = []
    ecd = []
    for j in range (len(Xtrain)):
        a1 = Xtrain.iloc[j,:]
        a2 = Xtest.iloc[i,:]
        ecd = distance.euclidean(a1,a2)
        #cos1 = distance.cosine(a1,a2)
        fdist = ecd
        distm.append(fdist)
    distall.append(distm)
#print (distall)
final5all=[]
for i in range (len(distall)):
    final5 = sorted(range(len(distall[i])), key=lambda x: distall[i][x])[:5]
    final5all.append(final5)
    
print (final5all)

nearest52 = []

for i in range (len(final5all)):
    top5 = []
    for j in range (len(final5all[i])):
        m= final5all[i][j]
        top5.append(Ytrain[m])
    nearest52.append(top5)
print(nearest52)
        


# # Normalizing euclidean and cosine distance and adding them together to get a distance measure and getting the closest top 5 names.

# In[142]:


from scipy.spatial import distance
import numpy as np

distall = []
for i in range (len(Xtest)):
    distm = []
    cos = []
    ecd = []
    for j in range (len(Xtrain)):
        a1 = Xtrain.iloc[j,:]
        a2 = Xtest.iloc[i,:]
        ecd1 = distance.euclidean(a1,a2)
        cos1 = distance.cosine(a1,a2)
        ecd.append(ecd1)
        cos.append(cos1)
    norm1 = cos / np.linalg.norm(cos)
    norm2 = ecd / np.linalg.norm(ecd)
    distm = np.add(norm1,norm2)
    distall.append(distm)
#print (distall)
final5all=[]
for i in range (len(distall)):
    final5 = sorted(range(len(distall[i])), key=lambda x: distall[i][x])[:5]
    final5all.append(final5)
    
print (final5all)

nearest52 = []

for i in range (len(final5all)):
    top5 = []
    for j in range (len(final5all[i])):
        m= final5all[i][j]
        top5.append(Ytrain[m])
    nearest52.append(top5)
print(nearest52)
        


# # Extracting name features from Test dataset without redaction

# In[143]:


import glob
import io
import os
import pdb
import sys

import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk


from nltk.chunk import tree2conlltags
import re
import pandas as pd


abc = []
persons = []
nwordsp= []
totchp = []
firstword = []
secondword = []
thirdword = []
fourthword = []
def get_entity(text):
    """Prints the entity inside of the text."""
    
    BC = u'\u2588'  
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                #print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
                #print (len(chunk))
                nwordsp.append(len(chunk))
                s=""
                k=0
                for c in chunk.leaves():
                    #print(c[0])
                    s+=c[0]
                    s+=" "
                    #print(s)
                    if len(chunk) == 1:
                        firstword.append(len(c[0]))
                        secondword.append(0)
                        thirdword.append(0)
                        fourthword.append(0)
                    if len(chunk) == 2 and k==0:
                        firstword.append(len(c[0]))
                    if len(chunk) == 2 and k==1:
                        secondword.append(len(c[0]))
                        thirdword.append(0)
                        fourthword.append(0)
                    if len(chunk) == 3 and k==0:
                        firstword.append(len(c[0]))
                    if len(chunk) == 3 and k==1:
                        secondword.append(len(c[0]))
                    if len(chunk) == 3 and k==2:
                        thirdword.append(len(c[0]))
                        fourthword.append(0)
                    if len(chunk) == 4 and k==0:
                        firstword.append(len(c[0]))
                    if len(chunk) == 4 and k==1:
                        secondword.append(len(c[0]))
                    if len(chunk) == 4 and k==2:
                        thirdword.append(len(c[0]))
                    if len(chunk) == 4 and k==3:
                        fourthword.append(len(c[0]))
                    k=k+1
                    
                    abc.append(c[0])
                persons.append(s[:-1])
                #print(len(s[:-1]))
                totchp.append(len(s[:-1]))  
#    print (persons)
#    print(len(persons))
#    for i in abc:
#        if i in text:
#            l=len(i)
#            text = text.replace(i,BC*l) 
#    print (text)
#    print (nwordsp)
#    print (totchp)
#    print (abc)
#    print (firstword)
#    print(len(firstword))
#    print (secondword)
#    print (thirdword)

    
def doextraction(glob_text):
    """Get all the files from the given glob and pass them to the extractor."""
    for thefile in glob.glob(glob_text):
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            get_entity(text)


 
#if __name__ == '__main__':
#    # Usage: python3 entity-extractor.py 'train/pos/*.txt'
doextraction('test/*.txt')


# # Getting the extracted dataset into a dataframe

# In[144]:


ytest=pd.DataFrame(persons, columns=['Persons'])
xtest = pd.DataFrame(nwordsp, columns=['count'])
xtest['totalch']= totchp
xtest['first']= firstword
xtest['sec']= secondword
xtest['third']= thirdword
xtest['fourth']= fourthword

print(xtest)
print (ytest)


# # Applying the Naive Bayes model on the test dataset and predicting names for test data

# In[145]:


Xtest=xtest.iloc[:,0:5]
Ytest=ytest.iloc[:,0]


result = model1.predict(Xtest)
print(result)
from sklearn.metrics import accuracy_score
accuracy_score(result, Ytest)


# # Applying kmeans to the test dataset and predicting names for test data

# In[146]:


ypred = model2.predict(Xtest)
#print (knn.kneighbors(Xtest)[1])
print(ypred)
from sklearn.metrics import accuracy_score
accuracy_score(ypred, Ytest)

