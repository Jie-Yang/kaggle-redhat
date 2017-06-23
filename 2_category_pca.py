# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import utils
import time

#%
column_names = utils.read_variable('outputs/column_names')
people_act_train = utils.read_variable('outputs/people_act_train')

#% convert category column to integer
#-----------------------------------
people_act_train_category2int = pd.DataFrame()
for name in column_names['category']:
    people_act_train_category2int[name] = people_act_train[name].str.replace('((type)|(group))\s','')

people_act_train_category2int = people_act_train_category2int.fillna(value=0)
del name
#% convert integer to one hot codes
one_hot_encoder = OneHotEncoder(n_values='auto', sparse=True)
one_hot_encoder.fit(people_act_train_category2int)

#% this variable will not presented in variable explorer and can NOT call toarray() which lead MemoryError
people_act_train_category_dummys = one_hot_encoder.transform(people_act_train_category2int)
print(people_act_train_category_dummys.shape)
print(type(people_act_train_category_dummys))

del people_act_train_category2int

#%%#####################################
# PCA
########################################

#%% can not use all data for PCA training which will lead to memoryEorror
########################################
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X, y = people_act_train_category_dummys, people_act_train[column_names['y']]

category_dummys_feature_filter = SelectKBest(chi2, k=X.shape[1])
category_dummys_feature_filter.fit(X, y)

#%
category_dummys_pvalues_significance_level = 0.05
# bigger p value, feature distribution is more like y
#normally set pvalues_threshold to 0.01 or 0.05
selected_col_ids = []
for id, p in enumerate(category_dummys_feature_filter.pvalues_):
    if p>category_dummys_pvalues_significance_level:
        selected_col_ids.append(id)
print(len(selected_col_ids))

utils.save_variable('outputs/people_act_train_category_selected_col_ids_pvalue_'+str(category_dummys_pvalues_significance_level),selected_col_ids)


del X,y,id, p

#%% filter on rows, based on 16GB RAW, it only can process 2K rows in a reasonable time.
Y = people_act_train[column_names['y']]

row_total_0 = 1000
row_total_1 = 1000
row_count_0 = 0
row_count_1 = 0
selected_row_ids = []
for i, y in enumerate(Y):
    if y==0 and row_count_0<row_total_0:
        selected_row_ids.append(i)
        row_count_0 = row_count_0+1
    elif y==1 and row_count_1<row_total_1:
        selected_row_ids.append(i)
        row_count_1 = row_count_1+1
    elif row_count_0==row_total_0 and row_count_0==row_total_0:
        break
print(len(selected_row_ids))

del Y,y,i
#%%
# Keep using csr_matrix. do NOT convert to DataFrame which will lead to Memory Error
people_act_train_category_dummys_filted = people_act_train_category_dummys[selected_row_ids,:]
people_act_train_category_dummys_filted = people_act_train_category_dummys_filted[:,selected_col_ids]
print(people_act_train_category_dummys_filted.shape)

#%%
import matplotlib.pyplot as plt

n_components=100

X = people_act_train_category_dummys_filted.toarray()
from sklearn.decomposition import PCA
startTime = time.time()
pca = PCA(n_components=n_components)
pca.fit(X)
print ('Training took', int(time.time() - startTime),'sec');
pca_variance = pca.explained_variance_ratio_

#%%
plt.figure(1) 
plt.title('Variances')
plt.plot(pca_variance)
plt.figure(2) 
plt.title('Accumulated Variance')
plt.plot(pca_variance.cumsum())
#%% 
pca_variance_theshold = 0.99
cs = pca_variance.cumsum()
for i in range(0,n_components,1):
    if cs[i]>pca_variance_theshold:
        break
n_components_opt = i+1
print('PC number for',pca_variance_theshold,':',n_components_opt,'=',cs[i])

#%% regenerate the PCA model with required pc nu.

X = people_act_train_category_dummys_filted.toarray()
from sklearn.decomposition import PCA
startTime = time.time()
pca_opt = PCA(n_components=n_components_opt)
pca_opt.fit(X)
print ('Training took', int(time.time() - startTime),'sec');
print('Accumulated Var:',sum(pca_opt.explained_variance_ratio_))
#%%#####################################
# persist result
########################################
utils.save_variable('outputs/people_act_train_category_pca_'+str(n_components_opt)+'pc',pca_opt)


#%%
X = people_act_train_category_dummys_filted.toarray()
X_new = pca_opt.transform(X)
print(X_new.shape)