import pandas as pd
import xgboost as xgb

# 读取数据预处理的训练集数据：33464*6746
df = pd.read_csv('G:\\ml360\\train\\1104.csv') # 我这里的1104.csv就是数据预处理后的训练数据集：33464*6749
df = df.drop(['loan_dt','id','tag'], axis=1)
train_x = df.ix[:,1:6746]
train_y = df.ix[:,[0]]

dtrain=xgb.DMatrix(train_x,label=train_y)

#booster:
params={'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':3,
        'lambda':1,
        'subsample':0.9,
        'colsample_bytree':0.5,
        'min_child_weight':3,
        'alpha':1e-5,
        'seed':0,
        'nthread':4,
        'silent':1,
        'gamma':0.4,
        'learning_rate' : 0.01} 
watchlist = [(dtrain,'train')]
bst = xgb.train(params,dtrain,num_boost_round=6000,evals=watchlist)
bst.save_model('G:\\ml360\\train\\test\\test_model3') # 保存实验模型
