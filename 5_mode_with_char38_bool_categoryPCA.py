import utils
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time
import sklearn.metrics as mx
#%%
people_act_train = utils.read_variable('outputs/people_act_train')
column_names = utils.read_variable('outputs/column_names')
people_act_train_category_dummys = utils.read_variable('outputs/people_act_train_category_dummys')
pc_number = 70
pca = utils.read_variable('outputs/people_act_train_category_pca_'+str(pc_number)+'pc')
selected_col_ids = utils.read_variable('outputs/people_act_train_category_selected_col_ids_pvalue_0.05')

from scipy.sparse import csr_matrix

# can NOT do transform all data in one process which will lead to memory leak

length = people_act_train_category_dummys.shape[0]

#%%########################
#% Use two IPython console and run the PCA asc and desc separately as a multiple-threading solution
##########################

#%% DSC####################
##########################
# can NOT do transform all data in one process which will lead to memory leak
people_act_train_category_pcs_dsc = csr_matrix((people_act_train_category_dummys.shape[0], pc_number))
#%% tolil() is more efficient than csr_matrix when values need to be modified
people_act_train_category_pcs_dsc_lil = people_act_train_category_pcs_dsc.tolil()
startTime = time.time()
for i in range(2174877,1160000,-1):
    people_act_train_category_pcs_dsc_lil[i] = pca.transform(people_act_train_category_dummys[i,selected_col_ids].toarray())
    print(i,'/',length)
print ('PCA Translation took', int(time.time() - startTime),'sec');

print(people_act_train_category_dummys.shape,'-PCA->',people_act_train_category_pcs_dsc_lil.shape)

#%
utils.save_variable('outputs/people_act_train_category_'+str(pc_number)+'pcs_dsc_lil',people_act_train_category_pcs_dsc_lil)

#%% ASC####################
##########################
# can NOT do transform all data in one process which will lead to memory leak
people_act_train_category_pcs = csr_matrix((people_act_train_category_dummys.shape[0], pc_number))
people_act_train_category_pcs_asc = people_act_train_category_pcs
#%%
people_act_train_category_pcs_asc_lil = people_act_train_category_pcs_asc.tolil()
startTime = time.time()
for i in range(0,1160001,1):
    people_act_train_category_pcs_asc_lil[i] = pca.transform(people_act_train_category_dummys[i,selected_col_ids].toarray())
    print(i,'/',length)
print ('PCA Translation took', int(time.time() - startTime),'sec');

print(people_act_train_category_dummys.shape,'-PCA->',people_act_train_category_pcs_asc_lil.shape)
#del people_act_train_category_dummys

#%
utils.save_variable('outputs/people_act_train_category_'+str(pc_number)+'pcs_asc_lil',people_act_train_category_pcs_asc_lil)


#%% Merge ASC and DSC result
temp_asc = utils.read_variable('outputs/people_act_train_category_'+str(pc_number)+'pcs_asc_lil')
temp_dsc = utils.read_variable('outputs/people_act_train_category_'+str(pc_number)+'pcs_dsc_lil')
 
# csr format is more effective than lil in arithmetic operations like +
people_act_train_category_pcs =  temp_asc.tocsr()+temp_dsc.tocsr()
utils.save_variable('outputs/people_act_train_category_'+str(pc_number)+'pcs',people_act_train_category_pcs)

del temp_asc,temp_dsc

#%%########################
#% Load PCs from provious processing
##########################
#% Test the backup of PCs
people_act_train_category_pcs = utils.read_variable('outputs/people_act_train_category_70pcs')

#% check data integration
length = people_act_train_category_pcs.shape[0]
c=0
for i in range(0,length,1):
    if people_act_train_category_pcs[i,0]==0:
        c=c+1
        print(i,'... ...',c)

print('missing rows:',c)
del c,i, length
#%%########################
#% Merge Y with X: PCs with char_38 and bool
##########################
cols = ['char_38']+column_names['bool']+['outcome']
pcs = pd.DataFrame(people_act_train_category_pcs.toarray())
train_merged = pd.concat([people_act_train[cols],pcs],axis=1)
print(train_merged.shape)
utils.save_variable('outputs/people_act_train_char38_bool_categorypc70',train_merged)
#%%########################
#% Get Validation dataset
##########################
people_act_train_char38_bool_categorypc70= utils.read_variable('outputs/people_act_train_char38_bool_categorypc70')

#%
train_grouped = people_act_train_char38_bool_categorypc70.groupby('outcome')
train_y0 = train_grouped.get_group(0)
train_y1 = train_grouped.get_group(1)

ratio_val_of_train = 0.1 

val = pd.concat([train_y0[0:int(train_y0.shape[0]*ratio_val_of_train)],train_y1[0:int(train_y1.shape[0]*ratio_val_of_train)]])
tr = pd.concat([train_y0[int(train_y0.shape[0]*ratio_val_of_train):],train_y1[int(train_y1.shape[0]*ratio_val_of_train):]])


#%%#####################################
# Model with only PCs
########################################

X= tr.drop(['outcome','char_38']+column_names['bool'],1)
Y = tr['outcome']
startTime = time.time()
model_char38_bool = tree.DecisionTreeClassifier(criterion='entropy').fit(X,Y)
print ('Training took',int(time.time() - startTime),'sec');
print('############ Only Category PCs ### Tr #############')
f_char38_bool = model_char38_bool.predict(X)
(f1_char38_bool,auc_char38_bool, confusion_char38_bool) = utils.validate_prediction(f_char38_bool,Y)

print('############ Only Category PCs ### Val ##############')
X = val.drop(['outcome','char_38']+column_names['bool'],1)
Y = val['outcome']
f_char38_bool_val = model_char38_bool.predict(X)
(f1_char38_bool_val,auc_char38_bool_val, confusion_char38_bool_val) = utils.validate_prediction(f_char38_bool_val,Y)


#%%#####################################
# Model with char 38 and bool and PC
########################################
X= tr.drop(['outcome'],1)
Y = tr['outcome']
startTime = time.time()
model_char38_bool_pcs = tree.DecisionTreeClassifier(criterion='entropy').fit(X,Y)
print ('Training took',int(time.time() - startTime),'sec');
print('############ Category PCs & char 38 & bool ### Tr #############')
f_char38_bool_pcs = model_char38_bool_pcs.predict(X)
utils.validate_prediction(f_char38_bool_pcs,Y)

print('############ Category PCs & char 38 & bool ### Val ##############')
X = val.drop(['outcome'],1)
Y = val['outcome']
f_char38_bool_val_pcs = model_char38_bool_pcs.predict(X)
utils.validate_prediction(f_char38_bool_val_pcs,Y)


# char_38 plus bool features improve the ROC from 84%(char_38) to 88% (char_38+bool)


#%%#####################################
# PCA on, char_38 bool, category 70 PCs for ploting
########################################
X= tr.drop(['outcome'],1)
Y = tr['outcome']
from sklearn.decomposition import PCA
startTime = time.time()
pca_X = PCA(n_components=X.shape[1])
pca_X.fit(X)
print ('Training took', int(time.time() - startTime),'sec');
pca_variance = pca_X.explained_variance_ratio_

#%%
import matplotlib.pyplot as plt
plt.figure(1) 
plt.title('Variances')
plt.plot(pca_variance)
plt.figure(2) 
plt.title('Accumulated Variance')
plt.plot(pca_variance.cumsum())
# the first 2 pcs carry 99.86% variances
#%%
X_pcs = pca_X.transform(X)

x_00 = pd.DataFrame(columns=['pc1','pc2'])
x_11 = pd.DataFrame(columns=['pc1','pc2'])
x_01 = pd.DataFrame(columns=['pc1','pc2'])
x_10 = pd.DataFrame(columns=['pc1','pc2'])
for i in range(0,X_pcs.shape[0],1):
    y = Y.iloc[i]
    f = f_char38_bool_pcs[i]
    pcs = pd.DataFrame([X_pcs[i,0:2]],columns=['pc1','pc2'])
    if y==0 and f==0:
        x_00 = x_00.append(pcs)
    elif y==1 and f==1:
        x_11 = x_11.append(pcs)
    elif y==0 and f==1:
        x_01 = x_01.append(pcs)
    elif y==1 and f==0:
        x_10 = x_10.append(pcs)
        
print('y==0,f==0:',len(x_00))
print('y==1,f==1:',len(x_11))
print('y==0,f==1:',len(x_01))
print('y==1,f==0:',len(x_10))
#%% Plot the results
plt.figure()
s = 50
plt.figure(1) 
plt.scatter(x_00['pc1'], x_00['pc2'], c="navy", s=s, label="y0,f0")
plt.legend()
plt.figure(2) 
plt.scatter(x_11['pc1'], x_11['pc2'], c="cornflowerblue", s=s, label="y1,f1")
plt.legend()
plt.figure(3) 
plt.scatter(x_01['pc1'], x_01['pc2'], c="c", s=s, label="y0,f1")
plt.legend()
plt.figure(4) 
plt.scatter(x_10['pc1'], x_10['pc2'], c="orange", s=s, label="y1,f0")
#plt.xlim([-6, 6])
#plt.ylim([-6, 6])
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("PC Plot of PCA on Decision Tree Classification Result")
plt.legend()
plt.show()

#%%
from sklearn.neighbors import KNeighborsClassifier
X= tr.drop(['outcome'],1)
Y = tr['outcome']
X_pcs = pca_X.transform(X)
X_knn = X_pcs[:,0:2]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_knn,Y)