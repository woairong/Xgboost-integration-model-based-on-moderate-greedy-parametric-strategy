import pandas as pd
import xgboost as xgb

# 读取赛题发放的测试集
test = pd.read_csv('G:\\ml360\\train\\test\\test(p).csv') # test(p).csv是赛题发放的测试集数据
id = test['id']
test = test.ix[:, 2:6747]

dtest = xgb.DMatrix(test)

bst1 = xgb.Booster(model_file='G:/ml360/train/test/test_model1')
bst2 = xgb.Booster(model_file='G:/ml360/train/test/test_model2')
bst3 = xgb.Booster(model_file='G:/ml360/train/test/test_model3')
bst4 = xgb.Booster(model_file='G:/ml360/train/test/test_model4')
bst5 = xgb.Booster(model_file='G:/ml360/train/test/test_model5')
bst6 = xgb.Booster(model_file='G:/ml360/train/test/test_model6')
bst7 = xgb.Booster(model_file='G:/ml360/train/test/test_model7')
bst8 = xgb.Booster(model_file='G:/ml360/train/test/test_model8')

ypred1 = bst1.predict(dtest)
ypred2 = bst2.predict(dtest)
ypred3 = bst3.predict(dtest)
ypred4 = bst4.predict(dtest)
ypred5 = bst5.predict(dtest)
ypred6 = bst6.predict(dtest)
ypred7 = bst7.predict(dtest)
ypred8 = bst8.predict(dtest)

ypred = 0.296*ypred1 + 0.148*ypred2 + 0.148*ypred3 + 0.074*ypred4 + 0.148*ypred5 + 0.074*ypred6 + 0.074*ypred7 + 0.038*ypred8

# 保存数据
ypred = list(ypred)
pd_ypred = pd.DataFrame(ypred, columns=['prob'])
id = pd.DataFrame([id])
id = id.T
result = pd.concat([id, pd_ypred], axis = 1)
result.to_csv('G:\\ml360\\train\\test\\result.txt', index = False, sep = ',')