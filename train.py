import os, sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold 
import matplotlib.pyplot as plt 
import numpy as np
import random

def load_data(path = 'german_credit_data.csv'):
    df_credit = pd.read_csv(path,index_col=0)
    #print(df_credit.info())
    #print('---------------------')
    #print(df_credit.head())
    return df_credit

def process_data(df_credit, col):
    df_credit = df_credit.merge(pd.get_dummies(df_credit[col], prefix=col), left_index=True, right_index=True)
    df_credit = df_credit.drop(col, axis = 1)
    
    return df_credit

def train_model(X, y):
    cv = StratifiedKFold(n_splits=2)
    model = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

    probas_ = model.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))
    print('auc:', roc_auc)
    probas_ = model.fit(X_train, y_train).predict(X_test)
    roc_auc = accuracy_score(y_test, probas_)
    print('acc:', roc_auc)
        
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.savefig('roc.png')

    return model.fit(X_train, y_train)

df_credit = load_data()
for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']:
    df_credit = process_data(df_credit, col)
#print('---------------------')
print(df_credit.head())
print(df_credit[(df_credit.Sex_female == 1) & (df_credit.Risk_bad == 1)].Sex_female.count())
print(df_credit[(df_credit.Sex_female == 0) & (df_credit.Risk_bad == 1)].Sex_female.count())
print(df_credit[(df_credit.Sex_female == 1) & (df_credit.Risk_bad == 0)].Sex_female.count())
print(df_credit[(df_credit.Sex_female == 0) & (df_credit.Risk_bad == 0)].Sex_female.count())

X = df_credit.drop(['Risk_bad', 'Risk_good'], 1).values
y = df_credit["Risk_bad"].values

model = train_model(X, y)
