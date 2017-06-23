# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import utils

#%%
column_names = utils.read_variable('outputs/column_names')
people_act_train = utils.read_variable('outputs/people_act_train')
#%%  Pre-process of Date columns
#-----------------------------------
#%% convert one date variable to three variables: year, month, and day
people_act_train_date_dummys = pd.DataFrame()
for name in column_names['date']:
    people_act_train_date_dummys[name+'_y'] = pd.to_datetime(people_act_train[name]).apply(lambda x: x.year)
    people_act_train_date_dummys[name+'_m'] = pd.to_datetime(people_act_train[name]).apply(lambda x: x.month)
    people_act_train_date_dummys[name+'_d'] = pd.to_datetime(people_act_train[name]).apply(lambda x: x.day)
del name
#%% convert category column to integer
#-----------------------------------
people_act_train_category2int = pd.DataFrame()
for name in column_names['category']:
    people_act_train_category2int[name] = people_act_train[name].str.replace('((type)|(group))\s','')

people_act_train_category2int = people_act_train_category2int.fillna(value=0)
del name
#%% convert integer to one hot codes
one_hot_encoder = OneHotEncoder(n_values='auto', sparse=True)
one_hot_encoder.fit(people_act_train_category2int)

#%% this variable will not presented in variable explorer and can NOT call toarray() which lead MemoryError
people_act_train_category_dummys = one_hot_encoder.transform(people_act_train_category2int)
print(people_act_train_category_dummys.shape)
print(type(people_act_train_category_dummys))

del people_act_train_category2int
#%%
utils.save_variable('outputs/people_act_train_date_dummys',people_act_train_date_dummys)
utils.save_variable('outputs/people_act_train_category_dummys',people_act_train_category_dummys)


