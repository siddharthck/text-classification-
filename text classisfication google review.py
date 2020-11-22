#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      crocodile
#
# Created:     24-04-2019
# Copyright:   (c) crocodile 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("/home/sid/PA"))  #C:\Users\crocodile

# Any results you write to the current directory are saved as output.
#Importing the Dataset
df =pd.read_csv("/home/sid/PA/googleplaystore_user_reviews.csv",encoding="latin1")


#Now Lets set dataset which collumns we are interested
df = pd.concat([df.Translated_Review, df.Sentiment], axis = 1)

#Now eleminate the nan value becasue they can affect our model
df.dropna(axis = 0, inplace = True)

df.Sentiment = [0 if i=="Positive" else 1 if i== "Negative" else 2 for i in df.Sentiment]

#Now lets Cleaning the Text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
text_list = []
for i in df.Translated_Review :
    review = re.sub('[^a-zA-Z]', ' ', i)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    stoplist=stopwords.words('english')
    review = [ps.stem(word) for word in review if not word in set(stoplist)]
    review = ' '.join(review)
    text_list.append(review)



# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
x = cv.fit_transform(text_list).toarray()
y = df.iloc[:, 1].values



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)


# Now Fitting Naive Bayes classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
#fit gaussianNB according to x,y
classifier.fit(x_train, y_train)


# Predicting the Test set results
"""
you can give any input later, so comenting right now!

y_pred = classifier.predict(x_test)
result_pred=[]
result_test=[]
for i in range (len(y_pred)):
	if y_pred[i]==0:
		result_pred.append("Positive")
	elif y_pred[i]==1:
		result_pred.append("Negative")
	elif y_pred[i]==2:
		result_pred.append("Neutral")

for i in range (len(y_test)):
	if y_test[i]==0:
		result_test.append("Positive")
	elif y_test[i]==1:
		result_test.append("Negative")
	elif y_test[i]==2:
		result_test.append("Neutral")


for i in range(len(y_test)):
	print result_test[i], "  ",result_pred[i],"\n"
	"""


# Making the Confusion Matrix and find Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(cm)
print(accuracy)
import seaborn as sn
import matplotlib.pyplot as plt
#plt.figure(figsize = (10,7))
df_cm = pd.DataFrame(cm, index=["Positive", "Negative", "Neutral"],columns=["Positive", "Negative", "Neutral"])
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 10})# font size
plt.title('Heat map for Naive Bayes classifier')

plt.show()

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, random_state = 0)
classifier.fit(x_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(x_test)



# Making the Confusion Matrix and find Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(cm)
print(accuracy)

df_cm = pd.DataFrame(cm, index=["Positive", "Negative", "Neutral"],columns=["Positive", "Negative", "Neutral"])
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 10})# font size
plt.title('Heat map for Random Forst classifier')

plt.show()
