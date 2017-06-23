import utils
from sklearn.ensemble import RandomForestClassifier
import time
#%%
tr= utils.read_variable('outputs/tr')
val= utils.read_variable('outputs/val')
#%%#####################################
# Training
########################################

X = tr.drop(['outcome'], axis=1)
Y = tr['outcome']

for estimator_nu in range(10, 11, 1):
    print('---------------')
    print('Tree Nu:',estimator_nu)
    model = RandomForestClassifier(n_estimators=estimator_nu, verbose=1, n_jobs=-1)

    startTime = time.time()
    model = model.fit(X, Y)
    print ('Training took', int(time.time() - startTime),'sec');

    utils.save_variable('models/model_estimator_nu_'+str(estimator_nu),model);

del X,Y
#%%#####################################
# Validation
########################################
for estimator_nu in range(10, 100, 10):
    print('###########################')
    print('Tree Nu:',estimator_nu)
    model = utils.read_variable('models/model_estimator_nu_'+str(estimator_nu))
    
    X = tr.drop(['outcome'], axis=1)
    Y = tr['outcome']
    
    f = model.predict(X)
    print('------TRAINING-------')    
    (tr_f1,tr_auc, tr_confusion) = utils.validate_prediction(f,Y)

    
    del X,Y,f
    #%
    X = val.drop(['outcome'], axis=1)
    Y = val['outcome']
    
    f = model.predict(X)
    print('------VALIDATION-------')    
    (val_f1,val_auc, val_confusion) = utils.validate_prediction(f,Y)
    
    print('============================')
    print('FINAL ==',val_auc)
    print('============================')
del X,Y,f

#%%#####################################
# model review
########################################
estimator_nu=2

print('###########################')
print('Tree Nu:',estimator_nu)
print('---------------------------')

X_col_exl = ['outcome','char_38']

X = tr.drop(X_col_exl, axis=1)
Y = tr['outcome']
# col char_38
model = RandomForestClassifier(n_estimators=estimator_nu, verbose=1, n_jobs=-1,max_features=1)

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

del X,Y,f

#%%#####################################
# Feature review
########################################
fi = model.feature_importances_
# based on the feature_importance, col char_38 give the most of importance, and prediction which only use char_38 also have good prediction result.