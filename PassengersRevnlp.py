#!/usr/bin/env python3
#

#importing all the necessay libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the sentiment dataset
#quoting =3 in this we will ingnore the double quotes ""
dataset =pd.read_csv('PassengersRev.tsv', delimiter = '\t', quoting=3)#delimeter daala and tab delimeter hai \t ka meaning hai tab


#Cleaning the text- this is because when we create bag of words model it will consist of only relevant words that the,on,and these are not relevant words because they won't help in the machine learning 
#Loved-Steming on positive words    disgrace-negative word
#Pre-Processing

import re  #cleaning tool
import nltk #package for nlp
nltk.download('stopwords')#package for removing the words
from nltk.corpus import stopwords #importing the above package
from nltk.stem.porter import PorterStemmer #stemming process replacing all words by root
#for all the reviews
corpus=[] #corpus collection of words

#iterating for all the dataset
for i in range(0,1000):
 
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #not removing any of the a-z letters and A-Z letters, remove question marks and characters
    #putting all the upper case into lower case
    review=review.lower()
    
    #removing all the irrelevant words that will not be meaningful for the bag model
    review = review.split()
    ps=PorterStemmer()
    review =[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    #now we convert the list into the string
    review=' '.join(review)
    
    corpus.append(review)
    
    
#creating bag of words models #creating sparse matrix
from sklearn.feature_extraction.text import CountVectorizer  #Countvector is class and then we will create obejct 
cv = CountVectorizer(max_features=1500)

#creating matrix of features
X = cv.fit_transform(corpus).toarray()#creating sparse matrix of columns and rows for prediction as defines in the copy
y = dataset.iloc[:,1].values  #creating our dependent variable columns Y= dependent variable

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix to optimize the prediction power and to see the number of correct predictions and errors 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(y_pred==0)
#accuracy
(54+87)/200