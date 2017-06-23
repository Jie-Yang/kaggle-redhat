import utils
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time
import sklearn.metrics as mx
#%%
people_act_train = utils.read_variable('outputs/people_act_train')
column_names = utils.read_variable('outputs/column_names')

#%%
#%% convert one date variable to three variables: year, month, and day
tr_with_date_dummys = people_act_train
for name in column_names['date']:
    tr_with_date_dummys[name+'_y'] = pd.to_datetime(people_act_train[name]).apply(lambda x: x.year)
    tr_with_date_dummys[name+'_m'] = pd.to_datetime(people_act_train[name]).apply(lambda x: x.month)
    tr_with_date_dummys[name+'_d'] = pd.to_datetime(people_act_train[name]).apply(lambda x: x.day)
del name


#%%

train_grouped = tr_with_date_dummys.groupby('outcome')
train_y0 = train_grouped.get_group(0)
train_y1 = train_grouped.get_group(1)

ratio_val_of_train = 0.1 

val = pd.concat([train_y0[0:int(train_y0.shape[0]*ratio_val_of_train)],train_y1[0:int(train_y1.shape[0]*ratio_val_of_train)]])
tr = pd.concat([train_y0[int(train_y0.shape[0]*ratio_val_of_train):],train_y1[int(train_y1.shape[0]*ratio_val_of_train):]])



#%%#####################################
# Model with date
########################################

date_dummy_cols = ['date_p_y','date_p_m','date_p_d','date_a_y','date_a_m','date_a_d']

X= tr[date_dummy_cols]
Y = tr['outcome']

model_date = tree.DecisionTreeClassifier().fit(X,Y)

f_date = model_date.predict(X)
print('############ date #################')
(f1_date,auc_date, confusion_date) = utils.validate_prediction(f_date,Y)

#%%#####################################
# Model with char 38 and date and bool
########################################
cols = ['char_38']+column_names['bool']+date_dummy_cols
X= tr[cols]
Y = tr['outcome']
startTime = time.time()
model_char38_bool = tree.DecisionTreeClassifier().fit(X,Y)
print ('Training took', int(time.time() - startTime),'sec');

print('############ char 38 & bool & date ### Tr #############')
f_char38_bool = model_char38_bool.predict(X)
(f1_char38_bool,auc_char38_bool, confusion_char38_bool) = utils.validate_prediction(f_char38_bool,Y)

print('############ char 38 & bool & date ### Val ##############')
X = val[cols]
Y = val['outcome']
f_char38_bool_val = model_char38_bool.predict(X)
(f1_char38_bool_val,auc_char38_bool_val, confusion_char38_bool_val) = utils.validate_prediction(f_char38_bool_val,Y)


# Observation: Overfitting problem: really high ROC (99%) for training, but low ROC (82%) for validation
# Overfitting means: we have good amount of features for the model. Next find the balance between Training and validation


#%%#####################################
# Fix OVERFITTING problem by config max_depth (ref scikit-learn decision tree doc)
########################################
cols = ['char_38']+column_names['bool']+date_dummy_cols

X_tr= tr[cols]
Y_tr = tr['outcome']
X_val = val[cols]
Y_val = val['outcome']
#%%
for max_depth in range(1, 1000, 1):
    #print('###########################')
    print('max_depth:',max_depth, end=',')

    startTime = time.time()
    model_char38_bool = tree.DecisionTreeClassifier(max_depth=max_depth).fit(X_tr,Y_tr)
    print (int(time.time() - startTime),'sec', end=',');
    
    f_tr = model_char38_bool.predict(X_tr)
    auc_tr = mx.roc_auc_score(f_tr,Y_tr)
    
    f_val = model_char38_bool.predict(X_val)
    auc_val = mx.roc_auc_score(f_val,Y_val)
    
    print('tr:',auc_tr,'val:',auc_val)

# observation: best val ROC (88%) is achieved with max_depth=1