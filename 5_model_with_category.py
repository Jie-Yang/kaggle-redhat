#%%
from sklearn import tree
import utils
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
import time
import sklearn.metrics as mx
#%%
people_act_train = utils.read_variable('outputs/people_act_train')
column_names = utils.read_variable('outputs/column_names')
#this variable will be hidden by variable explorer because it is not supported.
people_act_train_category_dummys = utils.read_variable('outputs/people_act_train_category_dummys')

print(people_act_train_category_dummys.shape)
print(type(people_act_train_category_dummys))
#%%#####################################
# feature selection of Category cols against char_38
########################################

X = people_act_train_category_dummys
Y_char38 = people_act_train['char_38']
Y_outcome = people_act_train['outcome']

kbest_char38 = SelectKBest(chi2, k=X.shape[1])
kbest_char38.fit(X, Y_char38)

kbest_outcome = SelectKBest(chi2, k=X.shape[1])
kbest_outcome.fit(X, Y_outcome)

#%%#####################################
# find top 100 features which are highly relevant to Y, but irrelevant/independent from char_38












########################################
p_char38 = pd.DataFrame(data={'p_value':kbest_char38.pvalues_})
p_char38_sorted = p_char38.sort_values(by='p_value',ascending=True)

p_outcome = pd.DataFrame(data={'p_value':kbest_outcome.pvalues_})
p_outcome_sorted = p_outcome.sort_values(by='p_value',ascending=False)
#%%

# bigger p value, feature distribution is more like y
#normally set pvalues_threshold to 0.01 or 0.05
feature_nu = len(kbest_char38.pvalues_)
selected_cols = pd.DataFrame(columns=['id','p_char38','p_outcome'])
for i in range(0,feature_nu,1):
    p_char38 = kbest_char38.pvalues_[i]
    p_outcome = kbest_outcome.pvalues_[i]
    if p_char38 < 0.01 and p_outcome < 0.9:
        new_row = pd.DataFrame.from_records([{'id':i,'p_char38':p_char38,'p_outcome':p_outcome}])
        selected_cols=selected_cols.append(new_row)
        print(len(selected_cols),'/',i,'/',feature_nu)
print(len(selected_cols),'/',i,'/',feature_nu)

del i, new_row, data, p_char38, p_outcome

selected_cols_sorted = selected_cols.sort_values(by='p_outcome',ascending=False)

#%%
selected_cols_ids = selected_cols_sorted[0:100]['id']
#%%
people_act_train_category_dummys_filted = people_act_train_category_dummys[:,selected_cols_ids]
print(people_act_train_category_dummys_filted.shape)


#%%#####################################
# Get Validation dataset
########################################

category_cols = pd.DataFrame(people_act_train_category_dummys_filted.toarray())
train_X = pd.concat([people_act_train['char_38'],
                     people_act_train[column_names['bool']],
                     category_cols],axis=1)
train_Y = people_act_train[column_names['y']]

train = pd.concat([train_X,train_Y],axis=1)
#%
train_grouped = train.groupby('outcome')
train_y0 = train_grouped.get_group(0)
train_y1 = train_grouped.get_group(1)

ratio_val_of_train = 0.1 

val = pd.concat([train_y0[0:int(train_y0.shape[0]*ratio_val_of_train)],train_y1[0:int(train_y1.shape[0]*ratio_val_of_train)]])
tr = pd.concat([train_y0[int(train_y0.shape[0]*ratio_val_of_train):],train_y1[int(train_y1.shape[0]*ratio_val_of_train):]])


#%%#####################################
# Modelling with Bool, Char_38, and filted category features
########################################

X_col_exl = ['outcome']
X_tr = tr.drop(X_col_exl, axis=1)
Y_tr = tr['outcome']
X_val = val.drop(X_col_exl, axis=1)
Y_val = val['outcome']
print('max_depth:','none', end='-->')
startTime = time.time()
model = tree.DecisionTreeClassifier().fit(X_tr,Y_tr)
print (int(time.time() - startTime),'sec', end=',');
#%
f_tr = model.predict(X_tr)
auc_tr = mx.roc_auc_score(f_tr,Y_tr) 
f_val = model.predict(X_val)
auc_val = mx.roc_auc_score(f_val,Y_val)   
print('tr:',auc_tr,',val:',auc_val)
#%%
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