import matplotlib.pyplot as plt
import utils
import pandas as pd

#%%
column_names = utils.read_variable('column_names')
#%%
#a_row = people[2:3]
#a_col = people['people_id']
#people_unique_ids= people.people_id.unique()
#a_cell = people[2:3]['people_id']


#%% join people and act tables
# right join: use act_train keys only
people_act_train = pd.merge(people, act_train, how='right',on='people_id',suffixes=('_p','_a'))
people_act_test = pd.merge(people, act_test, how='right',on='people_id',suffixes=('_p','_a'))

#%%
people_act_activity_category_type1 = people_act_train.loc[(people_act_train['activity_category']=='type 1')][0:1]
people_act_char_10_a_notnull = people_act_train.loc[people_act_train['char_10_a'].notnull() ][0:1]


#%%
people_act_grouped = people_act_train.groupby('outcome')
#%% outcome 0 VS. outcome 1
people_act_outcome0 = people_act_grouped.get_group(0)
people_act_outcome1 = people_act_grouped.get_group(1)


plt.pie([people_act_outcome0.shape[0],people_act_outcome1.shape[0]], labels=['0','1'],autopct='%1.4f%%' )
plt.show()

#%% review category variables
category_column_names_stat = pd.DataFrame(index=column_names['category'],columns=['unique_types'])
for name in column_names['category']:
    shape = people_act_train[name].unique().shape
    category_column_names_stat.ix[name].unique_types = shape[0]
print(category_column_names_stat)

#%%
people_act_train['group_1'].unique()
people_act_train['char_10_a'].unique()

#%%
utils.save_variable('people_act_train',people_act_train)
utils.save_variable('people_act_test',people_act_test)
