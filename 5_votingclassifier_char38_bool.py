import utils
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import pandas as pd
import time
import sklearn.metrics as mx
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
#%%
people_act_train = utils.read_variable('outputs/people_act_train')
column_names = utils.read_variable('outputs/column_names')

#%%
#%%

train_grouped = people_act_train.groupby('outcome')
train_y0 = train_grouped.get_group(0)
train_y1 = train_grouped.get_group(1)

ratio_val_of_train = 0.1 

val = pd.concat([train_y0[0:int(train_y0.shape[0]*ratio_val_of_train)],train_y1[0:int(train_y1.shape[0]*ratio_val_of_train)]])
tr = pd.concat([train_y0[int(train_y0.shape[0]*ratio_val_of_train):],train_y1[int(train_y1.shape[0]*ratio_val_of_train):]])

#%%#####################################
# Voting system
########################################
cols = ['char_38']+column_names['bool']
X= tr[cols]
Y = tr['outcome']

clf1 = SGDClassifier(loss="log", penalty="l2")
clf2 = tree.DecisionTreeClassifier()

model_voting = VotingClassifier(estimators=[('sgd', clf1), ('dt', clf2)],  voting='soft',weights=[1,2])
model_voting = model_voting.fit(X,Y)

print('############ Voting ### Tr #############')
f_voting_tr = model_voting.predict(X)
utils.validate_prediction(f_voting_tr,Y)

print('############ Voting ### Val ##############')
X = val[cols]
Y = val['outcome']
f_voting_val = model_voting.predict(X)
utils.validate_prediction(f_voting_val,Y)



#%%#####################################
# SGD
########################################
cols = ['char_38']+column_names['bool']
X= tr[cols]
Y = tr['outcome']
model_sgd = SGDClassifier(loss="hinge", penalty="l2").fit(X,Y)

print('############ SGW ### Tr #############')
f_sgd_tr = model_sgd.predict(X)
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
f1 = f_char38_bool_val
f2 = f_sgd_val
Y = val['outcome']
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
    elif Y.iloc[i]!=f1[i] and f1[i]==f2[i]:
        c3 = c3+1
    elif Y.iloc[i]==f1[i] and Y.iloc[i]==f2[i]:
        c4 = c4+1
        
print('RESULT:',c1,c2,c3,c4)
print('Can not be predicted by either M1 or M2:',str(c3/Y_len))

del f1,f2,Y,c1,c2,c3,c4,i

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