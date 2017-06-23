# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:40:13 2016

@author: jyang
"""
import pickle
import sklearn.metrics as mx

def save_variable(file_name,var):
    pkl_file = open(file_name+'.pkl', 'wb')
    pickle.dump(var, pkl_file, -1)
    pkl_file.close()
    
def read_variable(file_name):
    pkl_file = open(file_name+'.pkl', 'rb')
    var = pickle.load(pkl_file)
    pkl_file.close()
    return var

def validate_prediction(y,f):
    #best value at 1 and worst score at 0
    f1 = mx.f1_score(y,f)
    # statistic used by Kaggle
    auc = mx.roc_auc_score(y,f)
    confusion = mx.confusion_matrix(y,f)
    print('f1:',f1)
    print('AUC:',auc)
    print('conf:\n',confusion)
    return (f1,auc, confusion)
