# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense, BatchNormalization, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import initializers
from sklearn.metrics import roc_auc_score
import keras.backend as K
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

## File Input
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

sample_1 = train_data[train_data.target == 1]
sample_0 = train_data[train_data.target == 0]
for i in range(0,9):
    sample_0 = pd.concat([sample_0,sample_1])
sample_0 = sample_0.sample(frac=1.0, replace = True)
X_train, X_test, y_train, y_test = train_test_split(sample_0.drop(['ID_code','target'],axis = 1), 
                                                    sample_0.target,  test_size=0.50, random_state=9787)


## Logistic Regression (Optimal to use balanced 100,000 rows - insensitive to train/test split)
model_logit = LogisticRegression(class_weight='balanced')
model_logit.fit(X_train, y_train)
## Test Score
print(roc_auc_score(y_true = y_test, y_score = model_logit.predict_proba(X_test)[:,1]))
## Train Score
print(roc_auc_score(y_true = y_train, y_score = model_logit.predict_proba(X_train)[:,1]))


## Naive Bayes (Optimal to use balanced 100,000 rows - insensitive to train/test split)
model_NB = GaussianNB()
model_NB.fit(X_train, y_train)
## Test Score
print(roc_auc_score(y_true = y_test, y_score = model_NB.predict_proba(X_test)[:,1]))
## Train Score
print(roc_auc_score(y_true = y_train, y_score = model_NB.predict_proba(X_train)[:,1]))


## XG Boost (Optimal to use balanced 100,000 rows - insensitive to train/test split)
model_XGB = GradientBoostingClassifier(n_estimators=500, max_features=None)
model_XGB.fit(X_train,y_train)
## Test Score
print(roc_auc_score(y_true = y_test, y_score = model_XGB.predict_proba(X_test)[:,1]))
## Train Score
print(roc_auc_score(y_true = y_train, y_score = model_XGB.predict_proba(X_train)[:,1]))


new_df = pd.DataFrame({'NB':model_NB.predict_proba(X_test)[:,1], 'XGB': model_XGB.predict_proba(X_test)[:,1],
                       'logit':model_logit.predict_proba(X_test)[:,1], 'True_Labels':y_test})
new_df = pd.concat([X_test, new_df], axis = 1)

new_df['NB-error'] = abs(new_df['NB']-new_df['True_Labels'])
new_df['XGB-error'] = abs(new_df['XGB']-new_df['True_Labels'])
new_df['logit-error'] = abs(new_df['logit']-new_df['True_Labels'])
new_df['algo'] = np.zeros(new_df.shape[0])
for i in range(0,new_df.shape[0]):
        if((new_df['NB-error'].iloc[i] < new_df['XGB-error'].iloc[i]) & (new_df['NB-error'].iloc[i] < new_df['logit-error'].iloc[i])):
            new_df['algo'].iloc[i] = 'NB'
            continue
        if((new_df['XGB-error'].iloc[i] < new_df['NB-error'].iloc[i]) & (new_df['XGB-error'].iloc[i] < new_df['logit-error'].iloc[i])):
            new_df['algo'].iloc[i] = 'XGB'
            continue
        else:
            new_df['algo'].iloc[i] = "logit"

X_train_ps, X_test_ps, y_train_ps, y_test_ps = train_test_split(new_df.drop(['algo','True_Labels', 'NB-error','XGB-error','logit-error'], axis = 1), 
                                                                new_df['algo'], random_state = 13, test_size = 0.00000000001)
model_ensemble = GradientBoostingClassifier()
model_ensemble.fit(X_train_ps, y_train_ps)


## Submission Script
submission_stack = pd.DataFrame({'NB':model_NB.predict_proba(test_data.drop(['ID_code'], axis = 1))[:,1], 'XGB': model_XGB.predict_proba(test_data.drop(['ID_code'], axis = 1))[:,1],
                                'logit': model_logit.predict_proba(test_data.drop(['ID_code'], axis = 1))[:,1]})
submission_stack = pd.concat([test_data.drop(['ID_code'], axis=1), submission_stack],axis = 1, sort=False)
final_predictions = model_ensemble.predict(submission_stack)

for i in range(0,final_predictions.shape[0]):
    if (final_predictions[i] == 'NB'):
        final_predictions[i] = submission_stack['NB'].iloc[i]
        continue
    if (final_predictions[i] == 'logit'):
        final_predictions[i] = submission_stack['logit'].iloc[i]
        continue
    else:
        final_predictions[i] = submission_stack['XGB'].iloc[i]

submission = pd.DataFrame({'ID_code':test_data['ID_code'], 'target':final_predictions})
submission.to_csv('submission.csv', index=False)
