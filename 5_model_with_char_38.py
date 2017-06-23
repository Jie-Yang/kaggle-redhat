import utils
from sklearn.ensemble import RandomForestClassifier
import time
#%%
tr= utils.read_variable('outputs/tr')
val= utils.read_variable('outputs/val')
estimator_nu=2

print('###########################')
print('Char 38, Tree Nu:',estimator_nu)
print('---------------------------')

X_col_exl = ['outcome','char_38']

#X = tr.drop(X_col_exl, axis=1)
X= tr['char_38'].reshape(-1, 1)
Y = tr['outcome']
# col char_38
model = RandomForestClassifier(n_estimators=estimator_nu, verbose=0, n_jobs=-1)

startTime = time.time()
model = model.fit(X, Y)
print ('Training took', int(time.time() - startTime),'sec');

f = model.predict(X)
print('------TRAINING-------')
(tr_f1,tr_auc, tr_confusion) = utils.validate_prediction(f,Y)

del X,Y,f

#%
#X = val.drop(X_col_exl, axis=1)
X= val['char_38'].reshape(-1, 1)
Y = val['outcome']
f = model.predict(X)
print('------VALIDATION-------')
(val_f1,val_auc, val_confusion) = utils.validate_prediction(f,Y)


print('============================')
print('FINAL ==',val_auc)
print('============================')

model_char38 = model
utils.save_variable('models/model_char38',model_char38)
del model

del X,Y,f

print('###########################')
print('without Char 38, Tree Nu:',estimator_nu)
print('---------------------------')

X_col_exl = ['outcome','char_38']
X = tr.drop(X_col_exl, axis=1)
Y = tr['outcome']
# col char_38
model = RandomForestClassifier(n_estimators=estimator_nu, verbose=0, n_jobs=-1)

startTime = time.time()
model = model.fit(X, Y)
print ('Training took', int(time.time() - startTime),'sec');

f = model.predict(X)
print('------TRAINING-------')
(tr_f1,tr_auc, tr_confusion) = utils.validate_prediction(f,Y)

del X,Y,f

#%
X = val.drop(X_col_exl, axis=1)
Y = val['outcome']
f = model.predict(X)
print('------VALIDATION-------')
(val_f1,val_auc, val_confusion) = utils.validate_prediction(f,Y)


print('============================')
print('FINAL ==',val_auc)
print('============================')

model_without_char38 = model
utils.save_variable('models/model_without_char38',model_without_char38)

del model

del X,Y,f

#%%#####################################
# Combine model_char38 and model_without_char38
########################################
Y = tr['outcome']
# model_char38
X1= tr['char_38'].reshape(-1, 1)
f1 = model_char38.predict(X1)
print('-------char 38--------------')
(m1_f1,m1_auc, m1_confusion) = utils.validate_prediction(f1,Y)


# model_without_char38
X2 = tr.drop(['outcome','char_38'], axis=1)
f2 = model_without_char38.predict(X2)
print('-------WITHOUT char 38--------------')
(m2_f1,m2_auc, m2_confusion) = utils.validate_prediction(f2,Y)

#%%
c1=0
c2=0
c3=0
c4=0
Y_len = len(Y)
for i in range(0,Y_len):
    #print(i,Y_len)
    if Y.iloc[i]==f1[i] and Y.iloc[i]!=f2[i]:
        c1 = c1+1
    elif Y.iloc[i]!=f1[i] and Y.iloc[i]==f2[i]:
        c2 = c2+1
    elif Y.iloc[i]!=f1[i] and f1[i]==f2[i]:
        c3 = c3+1
    elif Y.iloc[i]==f1[i] and Y.iloc[i]==f2[i]:
        c4 = c4+1
        
print('RESULT:',c1,c2,c3,c4)
print('Can not be predicted by either M1 or M2:',str(c3/Y_len))


