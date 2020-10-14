# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:50:43 2020

@author: Prince
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')
test['Claim'] = 'NA'

full = pd.concat([train, test], axis = 0, keys = ['train', 'test'])
full.isna().sum()
preview = full.describe()
correlations = full.corr() #no multi-collinearity

#garden <- assume no garden, hence 'O'
#building dimension <- mean
#date of occupancy <- KNNI imputer with Building Dimension
#GEo_COde <- most mode


full['Garden'] = full['Garden'].fillna('O')
full['Building Dimension'] = full['Building Dimension'].fillna(full['Building Dimension'].mean())
full.groupby('Geo_Code').size().sort_values(ascending = False) #replace missing with most common
full['Geo_Code'] = full['Geo_Code'].fillna('6088')

from sklearn.impute import KNNImputer #only words for numerical variables
imputer = KNNImputer(n_neighbors = 3)
full['Date_of_Occupancy'] = imputer.fit_transform(full[['Date_of_Occupancy', 'Building Dimension']])


#clean up windows
cleaner_2 = lambda i: str(i).replace('>=', '')
cleaner = lambda i: float(str(i).replace('.', '0'))
full.iloc[:, -3] = full.iloc[:, -3].map(cleaner_2)
full.iloc[:, -3] = full.iloc[:, -3].map(cleaner)
full.iloc[:, [-7,-8,-9,-10]] = full.iloc[:, [-7,-8,-9,-10]].astype('category')

#full.groupby('Geo_Code').size()

full = full.drop(['Geo_Code', 'Customer Id' ], axis = 1)


plt.hist(full['Building Dimension'])  #right skewed hence need log transformation
plt.show()



plt.hist(full['NumberOfWindows'])
plt.show()

logging = lambda j: np.log10(j)  #normalizing
full['Building Dimension'] = full['Building Dimension'].map(logging)
full['duration_since_occupancy'] = full['YearOfObservation'] - full['Date_of_Occupancy']

#loggingwindows = lambda i: np.log10(i) if i != 0 else i

#full['NumberOfWindows'] = full['NumberOfWindows'].map(loggingwindows)



plt.hist(full['Date_of_Occupancy'])
plt.show() #bin by century? 

centuries = ['antic','early', 'modern', 'recent']
full['Date_of_Occupancy'] = pd.cut(full['Date_of_Occupancy'], bins = [1545,1800, 1900, 2000, 2016], labels=centuries)
full = full.drop(['YearOfObservation'], axis = 1)
full['Date_of_Occupancy'] = full['Date_of_Occupancy'].fillna('recent')
y_train = full.iloc[:7160, 10].astype('int')

full = full.drop(['Claim'], axis = 1)
#onehot encoding

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2,3,4,5,8])], remainder='passthrough')
full = np.array(ct.fit_transform(full))

#scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
full[:, [14, 16, 17]] = sc.fit_transform(full[:, [14, 16, 17]])


x_train = full[:7160,]
x_test = full[7160:,]


#RandomForest, XGbosst, Neural Network

'''from sklearn.ensemble import RandomForestClassifier
Random_classifier = RandomForestClassifier(n_estimators = 50,
                                           criterion = 'entropy', 
                                           random_state = 0,
                                           max_depth = 10,
                                           min_samples_split = 2)


Random_classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred1 = Random_classifier.predict(x_test)


'''


 #Deep Learning
import tensorflow as tf
tf.__version__


# Adding the input layer and the first hidden layer
#activation function in a fully connected neural netwrok must be rectifier "relu
from keras.wrappers.scikit_learn import KerasClassifier #for cross validation
from sklearn.model_selection import GridSearchCV

def kerasclassifier():
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=32, activation='relu'))   #DENSE class? #hyperparameters,  units = how many neurons
    ann.add(tf.keras.layers.Dense(units=30, activation='relu'))  #relu because of linear combination of inputs
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #output should be one because its binary. If more than 1, say a,b,c we need 3
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #for binary classification, always binary_crossentropy, #for none,
    return ann


model = KerasClassifier(build_fn = kerasclassifier, epochs = 150, batch_size = 100)

# Training the ANN on the Training set
model.fit(x_train, y_train)


#GRID SEARCH
'''from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [100, 50],
               'max_depth': [15, 10],
               'min_samples_split': [3, 2]}]



from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = x_train, y = y_train, cv = 6)
accuracies.mean()
   
gs = GridSearchCV(estimator = Random_classifier,
                  param_grid = parameters,
                  scoring = 'accuracy',
                  cv = 6,
                  n_jobs = -1)

gs = gs.fit(x_train, y_train)
best_accuracy = gs.best_score_
best_params = gs.best_params_   
print(best_accuracy)'''


y_pred1 = model.predict(x_test)

test['Claim'] = y_pred1


test.to_csv('insurance.csv', index=False)








