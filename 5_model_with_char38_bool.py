import utils
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time
import sklearn.metrics as mx
#%
people_act_train = utils.read_variable('outputs/people_act_train')
column_names = utils.read_variable('outputs/column_names')

#%

train_grouped = people_act_train.groupby('outcome')
train_y0 = train_grouped.get_group(0)
train_y1 = train_grouped.get_group(1)

ratio_val_of_train = 0.1 

val = pd.concat([train_y0[0:int(train_y0.shape[0]*ratio_val_of_train)],train_y1[0:int(train_y1.shape[0]*ratio_val_of_train)]])
tr = pd.concat([train_y0[int(train_y0.shape[0]*ratio_val_of_train):],train_y1[int(train_y1.shape[0]*ratio_val_of_train):]])


#%

#%#####################################
# Model with all bool features
########################################
#X= tr[column_names['bool']]
#Y = tr['outcome']
#model_bool = tree.DecisionTreeClassifier().fit(X,Y)
#
#f_bool_proba = model_bool.predict_proba(X)
#f_bool = model_bool.predict(X)
#print('############ bool #################')
#(f1_bool,auc_bool, confusion_bool) = utils.validate_prediction(f_bool,Y)


#%#####################################
# Model with char 38
########################################
#X= tr['char_38'].reshape(-1, 1)
#Y = tr['outcome']
#model_char38 = tree.DecisionTreeClassifier()
#model_char38 = model_char38.fit(X,Y)
#
#f_char38_proba = model_char38.predict_proba(X)
#f_char38 = model_char38.predict(X)
#print('############ char 38 #################')
#(f1_char38,auc_char38, confusion_char38) = utils.validate_prediction(f_char38,Y)


#%#####################################
# Model with char 38 and bool
########################################
cols = ['char_38']+column_names['bool']
X= tr[cols]
Y = tr['outcome']
model_char38_bool = tree.DecisionTreeClassifier().fit(X,Y)

print('############ char 38 & bool ### Tr #############')
f_char38_bool = model_char38_bool.predict(X)
(f1_char38_bool,auc_char38_bool, confusion_char38_bool) = utils.validate_prediction(f_char38_bool,Y)

print('############ char 38 & bool ### Val ##############')
X = val[cols]
Y = val['outcome']
f_char38_bool_val = model_char38_bool.predict(X)
(f1_char38_bool_val,auc_char38_bool_val, confusion_char38_bool_val) = utils.validate_prediction(f_char38_bool_val,Y)


# char_38 plus bool features improve the ROC from 84%(char_38) to 88% (char_38+bool)


#%%#####################################
# Model with char 38 and bool: examine Max Depth
########################################
cols = ['char_38']+column_names['bool']
X_tr = tr[cols]
Y_tr = tr['outcome']
X_val = val[cols]
Y_val = val['outcome']
for max_depth in range(1,100,1):    
    print('max_depth:',max_depth, end='-->')
    startTime = time.time()
    model_char38_bool = tree.DecisionTreeClassifier(max_depth=max_depth).fit(X_tr,Y_tr)
    print (int(time.time() - startTime),'sec', end=',');
    f_tr = model_char38_bool.predict(X_tr)
    auc_tr = mx.roc_auc_score(f_tr,Y_tr) 
    f_val = model_char38_bool.predict(X_val)
    auc_val = mx.roc_auc_score(f_val,Y_val)   
    print('tr:',auc_tr,',val:',auc_val)

    
#%%#####################################
# Model with char 38 and bool: examine with RandomForestClassifer
# Observation: no significant improvement by using RandomDecisionTree compare to simple DecisionTree
########################################
cols = ['char_38']+column_names['bool']
X_tr= tr[cols]
Y_tr = tr['outcome']
X_val = val[cols]
Y_val = val['outcome']
for estimator_nu in range(1, 100, 2):
    print('###########################')
    print('Tree Nu:',estimator_nu)
    
    startTime = time.time()
    model = RandomForestClassifier(n_estimators=estimator_nu, verbose=0, n_jobs=-1)
    model = model.fit(X_tr,Y_tr)
    print ('Training took', int(time.time() - startTime),'sec');
    
    f_tr = model.predict(X_tr)
    print('------TRAINING-------')    
    (tr_f1,tr_auc, tr_confusion) = utils.validate_prediction(f_tr,Y_tr)
    
    f_val = model.predict(X_val)
    print('------VALIDATION-------')    
    (val_f1,val_auc, val_confusion) = utils.validate_prediction(f_val,Y_val)
    
    print('============================')
    print('FINAL ==',val_auc)
    print('============================')

#%%#####################################
# Identify potential Negative results from Positive ones based on X 
########################################
X= tr[cols]
Y = tr['outcome']
model_char38_bool = tree.DecisionTreeClassifier().fit(X,Y)

print('############ char 38 & bool ### Tr #############')
f_char38_bool = model_char38_bool.predict(X)
(f1_char38_bool,auc_char38_bool, confusion_char38_bool) = mx.accuracy_score(f_char38_bool,Y)

nagative_row_ids = []
for i in range(0,len(Y),1):
    if f_char38_bool[i]!=Y.iloc[i]:
        nagative_row_ids.append(i)
print(len(nagative_row_ids))

#%%
import numpy as np
V = np.ones(len(tr['outcome']))
V[nagative_row_ids]=0
#%%
X= tr[cols]
Y = V
model_negative_classfier = tree.DecisionTreeClassifier().fit(X,Y)

print('############ char 38 & bool ### Tr #############')
f_negative = model_negative_classfier.predict(X)
print('Accurancy:',mx.accuracy_score(Y,f_negative))
confusion = mx.confusion_matrix(Y,f_negative)
print('conf:\n',confusion)