import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics  
from sklearn.grid_search import GridSearchCV  # Perforing grid search

df = pd.read_csv('G:\\ml360\\train\\1104.csv')
data0 = df.drop(['id', 'loan_dt', 'tag'], axis=1)

# 数据筛选
data = data0
dataT = data.T
dataT.isnull().sum()
X = dataT.isnull().sum()  # X是缺失值的序号集
x = list()  # x是超过缺失值指标的序号值集
for i in range(len(X)):
    if X[i] < 600:# 值越大，留下的数据集越多;500-1354,600-2456,675-3313,700-3636
        x.append(i)
data = data.ix[x]

train = data
target = 'label'

def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          metrics='auc', 
                          early_stopping_rounds=early_stopping_rounds, 
                          show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['label'], eval_metric='auc')
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['label'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['label'], dtrain_predprob))

# Choose all predictors except target & IDcols
A = [x for x in train.columns if x not in [target]]
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=140,
    max_depth=6,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)
modelfit(xgb1, train, predictors)


# 先对 max_depth，min_child_weight两组参数进行调参
param_test1 = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_child_weight': [1, 2, 3, 4, 5, 6]
}
gsearch1 = GridSearchCV(estimator=XGBClassifier(
        learning_rate=0.1, 
        n_estimators=140, 
        max_depth=5,
        min_child_weight=1, 
        gamma=0, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective='binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27),
        param_grid=param_test1, 
        scoring='roc_auc', 
        n_jobs=4, 
        iid=False,
        cv=5)
gsearch1.fit(train[predictors], train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print("gsearch1.grid_scores_:", gsearch1.grid_scores_, 
      " gsearch1.best_params_:", gsearch1.best_params_,
      "gsearch1.best_score_:", gsearch1.best_score_)
# 最优 max_depth=3， min_child_weight=3，得分：0.78699
# 次优 max_depth=7， min_child_weight=4，得分：0.78245

# 在 max_depth=3，min_child_weight=3情况下调参 gamma
param_test2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=200, 
        max_depth=3,
        min_child_weight=3, 
        gamma=0, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
        param_grid = param_test2, 
        scoring='roc_auc',
        n_jobs=4,
        iid=False,
        cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
print("gsearch2.grid_scores_:", gsearch2.grid_scores_,
      " gsearch2.best_params_:", gsearch2.best_params_,
      "gsearch2.best_score_:", gsearch2.best_score_)
# 测试结果 最优：gamma=0.2
#         次优：gamma=0.4

# 下面在 max_depth=3，min_child_weight=3，gamma=0条件下，调整 subsample，colsample_bytree的参数
param_test3 = {
 'subsample':[i/10.0 for i in range(1,10,2)],
 'colsample_bytree':[i/10.0 for i in range(1,10,2)]
}
gsearch3= GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=200, 
        max_depth=3,
        min_child_weight=3, 
        gamma=0.2, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
        param_grid = param_test3, 
        scoring='roc_auc',
        n_jobs=4,
        iid=False, 
        cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
print("gsearch3.grid_scores_:", gsearch3.grid_scores_, 
      " gsearch3.best_params_:", gsearch3.best_params_,
      "gsearch3.best_score_:", gsearch3.best_score_)
# 结果 subsample=0.9.colsample_bytree=0.5

# 更加精确化,分度值0.05
param_test4 = {
 'subsample':[i/10.0 for i in range(8,10)],
 'colsample_bytree':[i/10.0 for i in range(4,7)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=200, 
        max_depth=3,
        min_child_weight=3, 
        gamma=0.2, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
        param_grid = param_test4, 
        scoring='roc_auc',
        n_jobs=4,
        iid=False, 
        cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
print("gsearch4.grid_scores_:", gsearch4.grid_scores_, 
      " gsearch4.best_params_:", gsearch4.best_params_,
      "gsearch4.best_score_:", gsearch4.best_score_)
# 结果 最优：subsample=0.9, colsample_bytree=0.8
#      次优：subsample=0.8, colsample_bytree=0.8
# 得到第一组参数：max_depth=3，min_child_weight=3，gamma=0.2,subsample=0.9,colsample_bytree=0.8
# 得到第二组参数：max_depth=3，min_child_weight=3，gamma=0.2,subsample=0.8,colsample_bytree=0.8  

#下面在max_depth=3，min_child_weight=3，gamma=0.4条件下，对subsample，colsample_bytree进行调参
param_test5 = {
 'subsample':[i/10.0 for i in range(1,10,2)],
 'colsample_bytree':[i/10.0 for i in range(1,10,2)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=200, 
        max_depth=7,
        min_child_weight=5, 
        gamma=0.3, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
        param_grid = param_test5, 
        scoring='roc_auc',
        n_jobs=4,
        iid=False, 
        cv=5)
gsearch5.fit(train[predictors],train[target])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
print("gsearch5.grid_scores_:", gsearch5.grid_scores_, 
      " gsearch5.best_params_:", gsearch5.best_params_,
      "gsearch5.best_score_:", gsearch5.best_score_)
# 测试结果 最优：subsample=0.9, colsample_bytree=0.5
#         次优：subsample=0.95, colsample_bytree=0.5

# 精确化
param_test6 = {
 'subsample':[i/10.0 for i in range(8,10)],
 'colsample_bytree':[i/10.0 for i in range(4,7)]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=200, 
        max_depth=3,
        min_child_weight=3, 
        gamma=0., 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
        param_grid = param_test6, 
        scoring='roc_auc',
        n_jobs=4,
        iid=False, 
        cv=5)
gsearch6.fit(train[predictors],train[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
print("gsearch6.grid_scores_:", gsearch6.grid_scores_, 
      " gsearch6.best_params_:", gsearch6.best_params_,
      "gsearch6.best_score_:", gsearch6.best_score_)
# 得到第三组参数：max_depth=3，min_child_weight=3，gamma=0.4,subsample=0.9,colsample_bytree=0.5
# 得到第四组参数：max_depth=3，min_child_weight=3，gamma=0.4,subsample=0.95,colsample_bytree=0.5

# 这里对max_depth=7，min_child_weight=4条件下，对gamma 进行调参
param_test7 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch7 = GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=200, 
        max_depth=7,
        min_child_weight=4, 
        gamma=0, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
        param_grid = param_test7, 
        scoring='roc_auc',
        n_jobs=4,
        iid=False,
        cv=5)
gsearch7.fit(train[predictors],train[target])
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_
print("gsearch7.grid_scores_:", gsearch7.grid_scores_,
      " gsearch7.best_params_:", gsearch7.best_params_,
      "gsearch7.best_score_:", gsearch7.best_score_)
# 结果 最优：gamma=0
#      次优：gamma=0.1

# 在max_depth=7,min_child_weight=4,gamma=0条件下，对subsample，colsample_bytree进行调参
param_test8 = {
 'subsample':[i/10.0 for i in range(1,10,2)],
 'colsample_bytree':[i/10.0 for i in range(1,10,2)]
}
gsearch8 = GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=177, 
        max_depth=7,
        min_child_weight=4, 
        gamma=0, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
        param_grid = param_test8, 
        scoring='roc_auc',
        n_jobs=4,
        iid=False, 
        cv=5)
gsearch8.fit(train[predictors],train[target])
gsearch8.grid_scores_, gsearch8.best_params_, gsearch8.best_score_
print("gsearch8.grid_scores_:", gsearch8.grid_scores_, 
      " gsearch8.best_params_:", gsearch8.best_params_,
      "gsearch8.best_score_:", gsearch8.best_score_)
# 测试结果 subsample=0.9   colsample_bytree=0.9

# 精确化
param_test9 = {
 'subsample':[0.85, 0.9, 0.95],
 'colsample_bytree':[0.85, 0.9, 0.95]
}
gsearch9 = GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=177, 
        max_depth=7,
        min_child_weight=4, 
        gamma=0, 
        subsample=0.9, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
        param_grid = param_test9, 
        scoring='roc_auc',
        n_jobs=4,
        iid=False, 
        cv=5)
gsearch9.fit(train[predictors],train[target])
gsearch9.grid_scores_, gsearch9.best_params_, gsearch9.best_score_
print("gsearch9.grid_scores_:", gsearch9.grid_scores_, 
      " gsearch9.best_params_:", gsearch9.best_params_,
      "gsearch9.best_score_:", gsearch9.best_score_)
# 结果 最优：subsample=0.9，colsample_bytree=0.95
#      次优：subsample=0.95， colsample_bytree=0.95
# 得到第五组参数：max_depth=7，min_child_weight=4，gamma=0,subsample=0.9,colsample_bytree=0.95
# 得到第六组参数：max_depth=7，min_child_weight=4，gamma=0,subsample=0.95,colsample_bytree=0.95

# 在max_depth=7,min_child_weight=4,gamma=0.1条件下，对subsample，colsample_bytree进行调参
param_test10 = {
 'subsample':[i/10.0 for i in range(1,10,2)],
 'colsample_bytree':[i/10.0 for i in range(1,10,2)]
}
gsearch10 = GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=177, 
        max_depth=7,
        min_child_weight=4, 
        gamma=0.1, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
        param_grid = param_test10, 
        scoring='roc_auc',
        n_jobs=4,
        iid=False, 
        cv=5)
gsearch10.fit(train[predictors],train[target])
gsearch10.grid_scores_, gsearch10.best_params_, gsearch10.best_score_
print("gsearch10.grid_scores_:", gsearch10.grid_scores_, 
      " gsearch10.best_params_:", gsearch10.best_params_,
      "gsearch10.best_score_:", gsearch10.best_score_)
# 测试结果 subsample=0.9   colsample_bytree=0.3

# 精确化
param_test11 = {
 'subsample':[0.85, 0.9, 0.95],
 'colsample_bytree':[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
 }
gsearch11 = GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=177, 
        max_depth=7,
        min_child_weight=4, 
        gamma=0.1, 
        subsample=0.9, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
        param_grid = param_test11, 
        scoring='roc_auc',
        n_jobs=4,
        iid=False, 
        cv=5)
gsearch11.fit(train[predictors],train[target])
gsearch11.grid_scores_, gsearch11.best_params_, gsearch11.best_score_
print("gsearch11.grid_scores_:", gsearch11.grid_scores_, 
      " gsearch11.best_params_:", gsearch11.best_params_,
      "gsearch11.best_score_:", gsearch11.best_score_)
# 结果 最优：subsample=0.9，colsample_bytree=0.15
#      次优：subsample=0.9， colsample_bytree=0.3
# 得到第七组参数：max_depth=7，min_child_weight=4，gamma=0.1,subsample=0.9,colsample_bytree=0.15
# 得到第八组参数：max_depth=7，min_child_weight=4，gamma=0.1,subsample=0.9,colsample_bytree=0.3


##################################################################################################
## 综上共得到了8组实验参数：

# 得到第一组参数：max_depth=3，min_child_weight=3，gamma=0.2,subsample=0.9,colsample_bytree=0.8
# 得到第二组参数：max_depth=3，min_child_weight=3，gamma=0.2,subsample=0.8,colsample_bytree=0.8  
# 得到第三组参数：max_depth=3，min_child_weight=3，gamma=0.4,subsample=0.9,colsample_bytree=0.5
# 得到第四组参数：max_depth=3，min_child_weight=3，gamma=0.4,subsample=0.95,colsample_bytree=0.5
# 得到第五组参数：max_depth=7，min_child_weight=4，gamma=0,subsample=0.9,colsample_bytree=0.95
# 得到第六组参数：max_depth=7，min_child_weight=4，gamma=0,subsample=0.95,colsample_bytree=0.95
# 得到第七组参数：max_depth=7，min_child_weight=4，gamma=0.1,subsample=0.9,colsample_bytree=0.15
# 得到第八组参数：max_depth=7，min_child_weight=4，gamma=0.1,subsample=0.9,colsample_bytree=0.3

####################################################################################################


# 下面在对reg_alpha,reg_lambda 进行调参
param_test12 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch12 = GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=177, 
        max_depth=3,
        min_child_weight=3, 
        gamma=0.2, 
        subsample=0.9, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
        param_grid = param_test12, 
        scoring='roc_auc',
        n_jobs=4,iid=False, cv=5)
gsearch12.fit(train[predictors],train[target])
gsearch12.grid_scores_, gsearch12.best_params_, gsearch12.best_score_
print("gsearch12.grid_scores_:", gsearch12.grid_scores_, 
      " gsearch12.best_params_:", gsearch12.best_params_,
      "gsearch12.best_score_:", gsearch12.best_score_)
# reg_alpha=0.00001继续调试

param_test13 = {
 'reg_alpha':[0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
}
gsearch13 = GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.05, 
        n_estimators=1000, 
        max_depth=3,
        min_child_weight=3, 
        gamma=0.2, 
        subsample=0.9, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
        param_grid = param_test13,
        scoring='roc_auc',
        n_jobs=4,
        iid=False,
        cv=5)
gsearch13.fit(train[predictors],train[target])
gsearch13.grid_scores_, gsearch13.best_params_, gsearch13.best_score_
print("gsearch13.grid_scores_:", gsearch13.grid_scores_, 
      " gsearch13.best_params_:", gsearch13.best_params_,
      "gsearch13.best_score_:", gsearch13.best_score_)
# 最终reg_alpha=1e-5


#下面对reg_lambda进行调参
param_test14 = {
 'reg_lambda':[0, 1e-6, 1e-5, 1e-4, 1e-3]
}
gsearch14 = GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=177, 
        max_depth=3,
        min_child_weight=3, 
        gamma=0.2, 
        reg_alpha=1e-5,
        subsample=0.9, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
        param_grid = param_test14, 
        scoring='roc_auc',
        n_jobs=4,
        iid=False,
        cv=5)
gsearch14.fit(train[predictors],train[target])
gsearch14.grid_scores_, gsearch14.best_params_, gsearch14.best_score_
print("gsearch14.grid_scores_:", gsearch14.grid_scores_, 
      " gsearch14.best_params_:", gsearch14.best_params_,
      "gsearch14.best_score_:", gsearch14.best_score_)
#reg_lambda=0.001

param_test15 = {
 'reg_lambda':[1, 10, 100]
}
gsearch15 = GridSearchCV(estimator = XGBClassifier(
        learning_rate =0.1, 
        n_estimators=177, 
        max_depth=3,
        min_child_weight=3, 
        gamma=0.2,  
        reg_alpha=1e-5,
        subsample=0.9, 
        colsample_bytree=0.8,
        objective= 'binary:logistic', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
        param_grid = param_test15, 
        scoring='roc_auc',
        n_jobs=4,
        iid=False, 
        cv=5)
gsearch15.fit(train[predictors],train[target])
gsearch15.grid_scores_, gsearch15.best_params_, gsearch15.best_score_
print("gsearch15.grid_scores_:", gsearch15.grid_scores_, 
      " gsearch15.best_params_:", gsearch15.best_params_,
      "gsearch15.best_score_:", gsearch15.best_score_)
#最终reg_lambda=1

# 最终正则化参数是：reg_alpha=1e-5, reg_lambda=1


