#%%
from sklearn.ensemble import RandomForestClassifier
import utils
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
#%
people_act_train_date_dummys = utils.read_variable('people_act_train_date_dummys')
people_act_train = utils.read_variable('people_act_train')
column_names = utils.read_variable('column_names')
#this variable will be hidden by variable explorer because it is not supported.
people_act_train_category_dummys = utils.read_variable('people_act_train_category_dummys')


print(people_act_train_date_dummys.shape)
print(type(people_act_train_date_dummys))
print(people_act_train_category_dummys.shape)
print(type(people_act_train_category_dummys))
print(people_act_train.shape)
print(type(people_act_train))
        
#%%#####################################
# feature selection of Category cols
########################################

X, y = people_act_train_category_dummys, people_act_train[column_names['y']]

category_dummys_feature_filter = SelectKBest(chi2, k=X.shape[1])
category_dummys_feature_filter.fit(X, y)

#%
category_dummys_pvalues_significance_level = 0.9
# bigger p value, feature distribution is more like y
#normally set pvalues_threshold to 0.01 or 0.05
category_dummys_selected_col_ids = []
for id, p in enumerate(category_dummys_feature_filter.pvalues_):
    if p>category_dummys_pvalues_significance_level:
        category_dummys_selected_col_ids.append(id)
print(len(category_dummys_selected_col_ids))

# Keep using csr_matrix. do NOT convert to DataFrame which will lead to Memory Error
people_act_train_category_dummys_filted = people_act_train_category_dummys[:,category_dummys_selected_col_ids]
print(people_act_train_category_dummys_filted.shape)

del X,y,id, p
#%%#####################################
# feature selection of bools cols
########################################

X, y = people_act_train[column_names['bool']], people_act_train[column_names['y']]

bool_feature_filter = SelectKBest(chi2, k=X.shape[1])
bool_feature_filter.fit(X, y)

#%
bool_pvalues_significance_level = 0.01
# bigger p value, feature distribution is more like y
#normally set pvalues_threshold to 0.01 or 0.05
bool_selected_col_names = []
names = people_act_train[column_names['bool']].columns
for id, p in enumerate(bool_feature_filter.pvalues_):
    if p>bool_pvalues_significance_level:
        bool_selected_col_names.append(names[id])
print(len(bool_selected_col_names))

people_act_train_bool_filted = people_act_train[bool_selected_col_names]
print(people_act_train_bool_filted.shape)
del X,y,id, p, names

#%%#####################################
# feature selection of date dummy cols
########################################

X, y = people_act_train_date_dummys, people_act_train[column_names['y']]

date_dummys_feature_filter = SelectKBest(chi2, k=X.shape[1])
date_dummys_feature_filter.fit(X, y)

#%
date_dummys_pvalues_significance_level = 0.01

date_dummys_selected_col_names = []
names = people_act_train_date_dummys.columns
for id, p in enumerate(date_dummys_feature_filter.pvalues_):
    if p>date_dummys_pvalues_significance_level:
        date_dummys_selected_col_names.append(names[id])
print(len(date_dummys_selected_col_names))

people_act_train_date_dummys_filted = people_act_train_date_dummys[date_dummys_selected_col_names]
print(people_act_train_date_dummys_filted.shape)
del X,y,id, p, names

#%%#####################################
# combine all pre-processed features into a new X
########################################
category_cols = pd.DataFrame(people_act_train_category_dummys_filted.toarray())
train_X = pd.concat([people_act_train[column_names['nu']],
                     people_act_train_date_dummys_filted,
                     people_act_train_bool_filted,
                     category_cols],axis=1)
train_Y = people_act_train[column_names['y']]

train = pd.concat([train_X,train_Y],axis=1)

print(train_X.shape)
del category_cols, train_X, train_Y
#%%
utils.save_variable('train',train);


#%% clean memory for further process, since train_X would kill 9 GB memory
del category_dummys_pvalues_significance_level, category_dummys_selected_col_ids, \
    date_dummys_pvalues_significance_level, date_dummys_selected_col_names,\
    people_act_train, people_act_train_bool_filted, people_act_train_date_dummys,\
    people_act_train_date_dummys_filted, bool_selected_col_names, \
    bool_pvalues_significance_level, column_names