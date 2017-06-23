import utils
import matplotlib.pyplot as plt
import pandas as pd

#%%
train = utils.read_variable('train')

#%%#####################################
# Separate Valdiation Dataset from Training
########################################
train_grouped = train.groupby('outcome')
train_y0 = train_grouped.get_group(0)
train_y1 = train_grouped.get_group(1)


plt.pie([train_y0.shape[0],train_y1.shape[0]], labels=['0','1'],autopct='%1.4f%%' )
plt.show()

#%%
ratio_val_of_train = 0.1 

val = pd.concat([train_y0[0:int(train_y0.shape[0]*ratio_val_of_train)],train_y1[0:int(train_y1.shape[0]*ratio_val_of_train)]])
tr = pd.concat([train_y0[int(train_y0.shape[0]*ratio_val_of_train):],train_y1[int(train_y1.shape[0]*ratio_val_of_train):]])

del train_grouped, train_y0, train_y1, train, ratio_val_of_train

#%%
utils.save_variable('tr',tr);
utils.save_variable('val',val);

del tr,val
#%%

#X = people_act_train_category_dummys
#Y = people_act_train[column_names['y']]
#clf = RandomForestClassifier()
#clf = clf.fit(X, Y)
#feature_importances = clf.feature_importances_
#print(feature_importances)# -*- coding: utf-8 -*-

