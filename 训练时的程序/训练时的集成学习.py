import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 读取数据预处理的训练集数据：33464*6746
df = pd.read_csv('G:\\ml360\\train\\1104.csv') # 我这里的1104.csv就是数据预处理后的训练数据集：33464*6749
data = df.ix[:,4:6749]
flag = df['label']
train_x, test_x, train_y, test_y = train_test_split(data, flag, test_size = 0.3, random_state=0) 

dtest = xgb.DMatrix(test_x)

bst1 = xgb.Booster(model_file='G:/ml360/train/test/model1')
bst2 = xgb.Booster(model_file='G:/ml360/train/test/model2')
bst3 = xgb.Booster(model_file='G:/ml360/train/test/model3')
bst4 = xgb.Booster(model_file='G:/ml360/train/test/model4')
bst5 = xgb.Booster(model_file='G:/ml360/train/test/model5')
bst6 = xgb.Booster(model_file='G:/ml360/train/test/model6')
bst7 = xgb.Booster(model_file='G:/ml360/train/test/model7')
bst8 = xgb.Booster(model_file='G:/ml360/train/test/model8')

ypred1 = bst1.predict(dtest)
ypred2 = bst2.predict(dtest)
ypred3 = bst3.predict(dtest)
ypred4 = bst4.predict(dtest)
ypred5 = bst5.predict(dtest)
ypred6 = bst6.predict(dtest)
ypred7 = bst7.predict(dtest)
ypred8 = bst8.predict(dtest)

ypred = 0.296*ypred1 + 0.148*ypred2 + 0.148*ypred3 + 0.074*ypred4 + 0.148*ypred5 + 0.074*ypred6 + 0.074*ypred7 + 0.038*ypred8
y_pred = (ypred >= 0.5)*1

from sklearn import metrics
print ('集成学习下的实验结果：')
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
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example()')
plt.legend(loc="lower right")
plt.show()