import utils
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
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

#%%#####################################
# Decision Tree
########################################
cols = ['char_38']+column_names['bool']
X= tr[cols]
Y = tr['outcome']
model_dt = tree.DecisionTreeClassifier().fit(X,Y)

print('############ char 38 & bool ### Tr #############')
f_dt_tr = model_dt.predict(X)
f_dt_tr_proba = model_dt.predict_proba(X)
utils.validate_prediction(f_dt_tr,Y)

print('############ char 38 & bool ### Val ##############')
X = val[cols]
Y = val['outcome']
f_dt_val = model_dt.predict(X)
utils.validate_prediction(f_dt_val,Y)



#%#####################################
# SGD
########################################
cols = ['char_38']+column_names['bool']
X= tr[cols]
Y = tr['outcome']
model_sgd = SGDClassifier(loss="log").fit(X,Y)

print('############ SGW ### Tr #############')
f_sgd_tr = model_sgd.predict(X)
f_sgd_tr_proba = model_sgd.predict_proba(X)
utils.validate_prediction(f_sgd_tr,Y)

print('############ SGW ### Val ##############')
X = val[cols]
Y = val['outcome']
f_sgd_val = model_sgd.predict(X)
utils.validate_prediction(f_sgd_val,Y)


# char_38 plus bool features improve the ROC from 84%(char_38) to 88% (char_38+bool)

    
#%%#####################################
# Cross-comparison of Decision Tree and SGW results
########################################
#%%
f1 = f_dt_tr
f2 = f_sgd_tr
Y = tr['outcome']
c1=0
c2=0
c3=0
c4=0
Y_len = len(Y)
for i in range(0,Y_len):
    #print(i,Y_len)
    if Y.iloc[i]==f1[i] and Y.iloc[i]!=f2[i]:
        c1 = c1+1
    elif Y.iloc[i]!=f1[i] and Y.iloc[i]==f2[i]:
        c2 = c2+1
        y =Y.iloc[i]
        print(y,'--> -dt:',f_dt_tr_proba[i][y],'+sgd:',f_sgd_tr_proba[i][y])
    elif Y.iloc[i]!=f1[i] and f1[i]==f2[i]:
        c3 = c3+1
    elif Y.iloc[i]==f1[i] and Y.iloc[i]==f2[i]:
        c4 = c4+1
        
print('RESULT:',c1,c2,c3,c4)
print('Can not be predicted by either M1 or M2:',str(c3/Y_len))

del f1,f2,Y,c1,c2,c3,c4,i

#%%
probas = pd.DataFrame(columns=['y','dt_0','dt_1','sgd_0','sgd_1'])
probas['y'] = tr['outcome']
probas['dt_0'] = f_dt_tr_proba[:,0]
probas['dt_1'] = f_dt_tr_proba[:,1]
probas['sgd_0'] = f_sgd_tr_proba[:,0]
probas['sgd_1'] = f_sgd_tr_proba[:,1]

#%%
from sklearn.neighbors import KNeighborsClassifier
X = probas.drop('y',1)
Y = probas['y']
model_probas = KNeighborsClassifier(n_neighbors=3,n_jobs=-1).fit(X,Y)
print('############ char 38 & bool ### Tr #############')
f_probas_tr = model_probas.predict(X)
utils.validate_prediction(f_probas_tr,Y)
#%%#####################################
# Model with SGD and DT
########################################

cols = ['char_38']+column_names['bool']
X= tr[cols]
X['sgd']=f_sgd_tr
Y = tr['outcome']
model_sgd_dt = tree.DecisionTreeClassifier().fit(X,Y)

print('############ SGW ### Tr #############')
f_sgd_dt_tr = model_sgd_dt.predict(X)
utils.validate_prediction(f_sgd_dt_tr,Y)

print('############ SGW ### Val ##############')
X = val[cols]
X['sgd']=f_sgd_val
Y = val['outcome']
f_sgd_dt_val = model_sgd_dt.predict(X)
utils.validate_prediction(f_sgd_dt_val,Y)