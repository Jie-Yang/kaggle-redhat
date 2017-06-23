import utils
from sklearn import tree
import time
import pandas as pd
#%%
tr= utils.read_variable('outputs/tr')
val= utils.read_variable('outputs/val')
estimator_nu=2

model_char38 = utils.read_variable('models/model_char38')
model_without_char38 = utils.read_variable('models/model_without_char38')

#%%#####################################
# Combine model_char38 and model_without_char38
########################################
Y = tr['outcome']
# model_char38
X1= tr['char_38'].reshape(-1, 1)
f1_proba = model_char38.predict_proba(X1)
f1 = model_char38.predict(X1)
print(model_char38.classes_)


# model_without_char38
X2 = tr.drop(['outcome','char_38'], axis=1)
f2_proba = model_without_char38.predict_proba(X2)
f2 = model_without_char38.predict(X2)
print(model_without_char38.classes_)


#%%
data = {'f1_0':f1_proba[:,0],'f1_1':f1_proba[:,1],'f2_0':f2_proba[:,0],'f2_1':f2_proba[:,1]}
f12_proba = pd.DataFrame(data=data, columns=['f1_0','f1_1','f2_0','f2_1'])
del data


#%%
X = f12_proba

ensemble_model = tree.DecisionTreeClassifier()

startTime = time.time()
ensemble_model = ensemble_model.fit(X, Y)
print ('Training took', int(time.time() - startTime),'sec');
f12 = ensemble_model.predict(X)
(f12_tr_f1,f12_tr_auc, f12_tr_confusion) = utils.validate_prediction(f12,Y)

Y = tr['outcome']
# model_char38
X1= tr['char_38'].reshape(-1, 1)
f1 = model_char38.predict(X1)
print('-------char 38--------------')
(m1_f1,m1_auc, m1_confusion) = utils.validate_prediction(f1,Y)

#%% CONCLUSION ######################
# features seleced by setting up following filters:
#    1. category features:p-value>0.9 (169 left)
#    2. bool features: p-value >0.01 (0 left)
#    3. date features: p-value > 0.01 (2 left)
# do not have significant improvement compared to model which only based on single feature char_38
# In other words, 
#    1.features which are ignored by these filters may bring useful information (e.g. bool features).
#    2. filted features except char_38 are not useful.

