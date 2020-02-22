from pylab import mpl 
import pandas as pd
from xgboost import plot_importance
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 读取数据预处理的训练集数据：33464*6746
df = pd.read_csv('G:\\ml360\\train\\1104.csv') # 我这里的1104.csv就是数据预处理后的训练数据集：33464*6749
data = df.ix[:,4:6749]
flag = df['label']
train_x, test_x, train_y, test_y = train_test_split(data, flag, test_size = 0.3, random_state=0) 

dtrain=xgb.DMatrix(train_x,label=train_y)
dtest = xgb.DMatrix(test_x)

#booster:
params={'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':3,
        'lambda':1,
        'subsample':0.9,
        'colsample_bytree':0.8,
        'min_child_weight':3,
        'alpha':1e-5,
        'seed':0,
        'nthread':4,
        'silent':1,
        'gamma':0.2,
        'learning_rate' : 0.01} 
watchlist = [(dtrain,'train')]
bst = xgb.train(params,dtrain,num_boost_round=5000,evals=watchlist)
bst.save_model('G:\\ml360\\train\\test\\model1') # 保存实验模型

ypred=bst.predict(dtest)
y_pred = (ypred >= 0.5)*1

# 画出特征得分图
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 800
plot_importance(bst) 

# 画出AUC
from sklearn import metrics
print ('参数模型1下的实验结果：')
print ('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
print ('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
print ('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
metrics.confusion_matrix(test_y,y_pred)
 
fpr,tpr,threshold = roc_curve(test_y, ypred) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example(参数模型1)')
plt.legend(loc="lower right")
plt.show()