# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:46:00 2016

@author: jyang
"""

# -*- coding: utf-8 -*-

import pandas as pd
import utils

act_train = pd.read_csv('data/act_train.csv')
act_test = pd.read_csv('data/act_test.csv')
people = pd.read_csv('data/people.csv')


#%%
print(people.columns.values)
print(act_train.columns.values)

#%%
column_names = {}
column_names['category']= ['char_1_p','group_1','char_2_p','char_3_p','char_4_p',
                         'char_5_p','char_6_p','char_7_p','char_8_p','char_9_p',
                         'activity_category','char_1_a','char_2_a','char_3_a'
                         ,'char_4_a','char_5_a','char_6_a','char_7_a','char_8_a'
                         ,'char_9_a','char_10_a']
column_names['date'] = ['date_p','date_a']
column_names['ignore'] = ['people_id','activity_id']
column_names['y'] = 'outcome'
column_names['bool'] = ['char_10_p','char_11','char_12','char_13','char_14',
                     'char_15','char_16','char_17','char_18','char_19',
                     'char_20','char_21','char_22','char_23','char_24',
                     'char_25','char_26','char_27','char_28','char_29',
                     'char_30','char_31','char_32','char_33','char_34',
                     'char_35','char_36','char_37']
column_names['nu'] = 'char_38'
utils.save_variable('column_names',column_names)