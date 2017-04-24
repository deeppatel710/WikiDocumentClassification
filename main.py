import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
import gensim
import numpy as np
import os
import time
start_time = time.time()

categories = ['art','people','culture','books','design','politics','technology','psychology',
'research','religion' ,'music','math','development','theory','philosophy','language','science','programming','history','software']

file_categories = []

file = open('doctag.txt','r')

with open('fileid.txt', 'r') as f:
	files = f.read()
file_id = files.split()

corpus = []

for fee in file_id:
	f = open(os.getcwd()+"/cleanFiles/"+fee + '.csv' , 'r')
	doc = f.read().split(',')
	corpus.append(doc)

print "corpus ready!"


kvPair = {}

for line in file:
	a = line

	b = a.split(":")
	c = b[1].strip()

	c = c[1:-1].replace(" ","").replace("'","").split(",")

	d =  list(set(categories).intersection(c))
	if d != []:
		kvPair[b[0].strip()] = d
	else:
		pass

for i in file_id:
	file_categories.append(kvPair[i])
print "fileCategories ready!"


train_files = file_id[:13116]
test_files = file_id[13116:]
train_categories = file_categories[:13116]
test_categories = file_categories[13116:]

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_categories)

y_test = mlb.fit_transform(test_categories)

# # Load Google's pre-trained Word2Vec model.
print "model loading ............."
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print "model loaded"

featureVec = [0]*len(train_files)

for i in xrange(len(train_files)):
	featureVec[i] = np.zeros(300,dtype="float32")
	nwords = 0

	for word in set(corpus[i]):
	    try:
	    	featureVec[i] = np.add(featureVec[i],model[word])
	    	nwords = nwords + 1
	    except KeyError:
	    	pass

	featureVec[i] = np.divide(featureVec[i],nwords)

	featureVec[i] = featureVec[i].reshape(1,-1)


featureVec = np.array([np.array(xi) for xi in featureVec])

print featureVec.shape



test_featureVec = [0]*len(test_files)

l = 13116
for i in xrange(len(test_files)):
	test_featureVec[i] = np.zeros(300,dtype="float32")
	nwords = 0

	for word in set(corpus[l]):
	    try:
	    	test_featureVec[i] = np.add(test_featureVec[i],model[word])
	    	nwords = nwords + 1
	    except KeyError:
	    	pass

	test_featureVec[i] = np.divide(test_featureVec[i],nwords)

	test_featureVec[i] = test_featureVec[i].reshape(1,-1)
	l+=1

test_featureVec = np.array([np.array(xi) for xi in test_featureVec])

print test_featureVec.shape


import sklearn
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


X = featureVec
X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))

X_test = test_featureVec
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))

print "Training classifier model....."
predictions = OneVsRestClassifier(LogisticRegression(random_state=0)).fit(X, y_train).predict(X_test)

# predictions = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y_train).predict(X_test)

# forest = RandomForestClassifier(n_estimators=100, random_state=1)
# multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
# predictions = multi_target_forest.fit(X, y_train).predict(X_test)


print "########## .CLASSIFICATION REPORT. ###########"
print(classification_report(y_test, predictions)) 


print "======Runtime=====\n", time.time() - start_time
