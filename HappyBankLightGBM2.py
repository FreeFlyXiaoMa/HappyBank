import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

#读取数据
train=pd.read_csv('Bank_FE_train.csv')
print(train.info())
y_train=train['Disbursed']
X_train=train.drop(['ID','Disbursed'],axis=1)
feat_names=X_train.columns
print(feat_names)

MAX_ROUNDS=10000
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=3)

#直接调用lgbm内核自带的cv方法
def get_n_estimators(params,X_train_1,y_trian_1,early_stopping_rounds=10):
    lgbm_params=params.copy()
   # y_trian_1.astype(float)
    dataset=lgbm.Dataset

    lgbmtrain=lgbm.Dataset(X_train_1,y_trian_1)
    cv_result=lgbm.cv(lgbm_params,lgbmtrain,num_boost_round=MAX_ROUNDS,early_stopping_rounds=early_stopping_rounds,seed=3,nfold=5,
                metrics='auc')
    #print('best n_estimators',len(cv_result['auc-mean']))
    #print('best cv score:',cv_result['auc-mean'][-1])
    return len(cv_result['auc-mean'])


params = {'boosting_type': 'goss',
          'objective': 'binary',
          'is_unbalance':True,
          #'categorical_feature': names:'City', 'Employer_Name'3, 'Salary_Account'15,'Device_Type'1,
          # 'Filled_Form'5,'Gender'6,'Mobile_Verified'12,'Source'16,'Var1'17,'Var2'18,'Var4'19,
          'categorical_feature': [0,1,3,5,6,12,15,16,17,18,19,20],
          #'categorical_feature':[0,3,15,1,5,6,12,16,17,18,19],
          'n_jobs': 4,
          'learning_rate': 0.1,
          #'n_estimators':n_estimators_1,
          'num_leaves': 60,
          'max_depth': 6,
          'colsample_bytree': 0.7,
          'verbosity':5
         }
n_estimators=get_n_estimators(params,X_train,y_train)
#best n_estimators 41
#best cv score: 0.823255900200085

#print(n_estimators)
params = {'boosting_type': 'goss',
          'objective': 'binary',
          'is_unbalance':True,
          'n_estimators':n_estimators,
          #'categorical_feature': names:'City', 'Employer_Name'3, 'Salary_Account'15,'Device_Type'1,
          # 'Filled_Form'5,'Gender'6,'Mobile_Verified'12,'Source'16,'Var1'17,'Var2'18,'Var4'19,
          'categorical_feature': [0,1,3,5,6,12,15,16,17,18,19,20],
          #'categorical_feature':[0,3,15,1,5,6,12,16,17,18,19],
          'n_jobs': 4,
          'learning_rate': 0.1,
          #'n_estimators':n_estimators_1,
          #'num_leaves': 60,
          'max_depth': 6,
          'colsample_bytree': 0.7,
          'verbosity':5
         }
if __name__=='__main__':
    n_leaves_s=range(30,90,10)
    lg=LGBMClassifier(silent=False,**params)
    tuned_parameter=dict(num_leaves=n_leaves_s)
    grid_search=GridSearchCV(lg,param_grid=tuned_parameter,cv=kfold,scoring='roc_auc',refit=False,
                     n_jobs=4)
    grid_search.fit(X_train,y_train)
    print('best score:',grid_search.best_score_)
    print('best params:',grid_search.best_params_)
    test_means=grid_search.cv_results_['mean_test_score']
    x_axis=n_leaves_s
#plt.plot(x_axis,test_means)
# plt.xlabel('num_leaves')
#plt.ylabel('AUC')
#plt.show()
#从cv曲线图中可以看出num_leaves=50时，AUV对应的分数最高
# best score: 0.8266261956699197
#best params: {'num_leaves': 50}

    #调整学习率
    params = {'boosting_type': 'goss',
              'objective': 'binary',
              'is_unbalance': True,
              'n_estimators': n_estimators,
              # 'categorical_feature': names:'City', 'Employer_Name'3, 'Salary_Account'15,'Device_Type'1,
              # 'Filled_Form'5,'Gender'6,'Mobile_Verified'12,'Source'16,'Var1'17,'Var2'18,'Var4'19,
              'categorical_feature': [0, 1, 3, 5, 6, 12, 15, 16, 17, 18, 19, 20],
              # 'categorical_feature':[0,3,15,1,5,6,12,16,17,18,19],
              'n_jobs': 4,
              #'learning_rate': 0.1,
              # 'n_estimators':n_estimators_1,
              'num_leaves': 50,
              'min_child_samples':20,
              'max_depth': 6,
              'colsample_bytree': 0.7,
              'verbosity': 5
              }
    tuned_parameter={'learning_rate':[0.0001,0.001,0.01,0.1,1,10,100]}
    lg=LGBMClassifier(silent=False,**params)
    grid_search=GridSearchCV(lg,n_jobs=4,param_grid=tuned_parameter,scoring='roc_auc',refit=False,cv=kfold)
    grid_search.fit(X_train,y_train)
    print('learning_rate best score:',grid_search.best_score_)
    print('learning_rate best params:',grid_search.best_params_)
    test_means=grid_search.cv_results_['mean_test_score']
    x_axis=[0.0001,0.001,0.01,0.1,1,10,100]
    plt.plot(x_axis,test_means)
    plt.xlabel('learning_rate')
    plt.ylabel('roc_auc')
    plt.show()

    #调整min_child_sample
    params = {'boosting_type': 'goss',
          'objective': 'binary',
          'is_unbalance': True,
          'n_estimators': n_estimators,
          # 'categorical_feature': names:'City', 'Employer_Name'3, 'Salary_Account'15,'Device_Type'1,
          # 'Filled_Form'5,'Gender'6,'Mobile_Verified'12,'Source'16,'Var1'17,'Var2'18,'Var4'19,
          'categorical_feature': [0, 1, 3, 5, 6, 12, 15, 16, 17, 18, 19, 20],
          # 'categorical_feature':[0,3,15,1,5,6,12,16,17,18,19],
          'n_jobs': 4,
          'learning_rate': 0.1,
          # 'n_estimators':n_estimators_1,
           'num_leaves': 50,
          'max_depth': 6,
          'colsample_bytree': 0.7,
          'verbosity': 5
          }

    min_child_samples_s=range(10,60,10)
    lg=LGBMClassifier(silent=False,**params)
    tuned_parameter=dict(min_child_samples=min_child_samples_s)
    grid_search=GridSearchCV(lg,n_jobs=4,param_grid=tuned_parameter,cv=kfold,refit=False,scoring='roc_auc',
                         )
    grid_search.fit(X_train,y_train)
    print('min_child_samples best score',grid_search.best_score_)
    print('min_child_samples best parms',grid_search.best_params_)
    test_means=grid_search.cv_results_['mean_test_score']
    x_axis=min_child_samples_s
# plt.plot(x_axis,test_means)
#plt.xlabel('min_child_samples')
#plt.ylabel('AUC')
#plt.show()
#min_child_samples best score 0.8266261956699197
#min_child_samples best parms {'min_child_samples': 20}

    #调整列采样比例
    params = {'boosting_type': 'goss',
              'objective': 'binary',
              'is_unbalance': True,
              'n_estimators': n_estimators,
              # 'categorical_feature': names:'City', 'Employer_Name'3, 'Salary_Account'15,'Device_Type'1,
              # 'Filled_Form'5,'Gender'6,'Mobile_Verified'12,'Source'16,'Var1'17,'Var2'18,'Var4'19,
              'categorical_feature': [0, 1, 3, 5, 6, 12, 15, 16, 17, 18, 19, 20],
              # 'categorical_feature':[0,3,15,1,5,6,12,16,17,18,19],
              'n_jobs': 4,
              'learning_rate': 0.1,
              # 'n_estimators':n_estimators_1,
              'num_leaves': 50,
              'min_child_samples':20,
              'max_depth': 6,
              #'colsample_bytree': 0.7,
              'verbosity': 5
              }
    lg=LGBMClassifier(silent=False,**params)
    colsample_bytree_s=[i/10.0 for i in range(5,10)]
    tuned_parameter=dict(colsample_bytree=colsample_bytree_s)
    grid_search=GridSearchCV(lg,n_jobs=4,param_grid=tuned_parameter,refit=False,scoring='roc_auc',cv=kfold)
    grid_search.fit(X_train,y_train)
    test_means=grid_search.cv_results_['mean_test_score']
    print('colsample_bytree best score:',grid_search.best_score_)
    print('colsample_bytree best param:',grid_search.best_params_)

    x_axis=colsample_bytree_s
    plt.plot(x_axis,test_means)
    plt.xlabel('colsample_bytree')
    plt.ylabel('AUC')
    plt.show()
    #colsample_bytree best score: 0.8266261956699197
    #colsample_bytree best param: {'colsample_bytree': 0.7}

    #得出结果列采样为0.7，方便起见行采样也选择0.7

    #采用以上参数总体训练一次
    params = {'boosting_type': 'goss',
              'objective': 'binary',
              'is_unbalance': True,
              'n_estimators': n_estimators,
              # 'categorical_feature': names:'City', 'Employer_Name'3, 'Salary_Account'15,'Device_Type'1,
              # 'Filled_Form'5,'Gender'6,'Mobile_Verified'12,'Source'16,'Var1'17,'Var2'18,'Var4'19,
              'categorical_feature': [0, 1, 3, 5, 6, 12, 15, 16, 17, 18, 19, 20],
              # 'categorical_feature':[0,3,15,1,5,6,12,16,17,18,19],
              'n_jobs': 4,
              'learning_rate': 0.01,
              # 'n_estimators':n_estimators_1,
              'num_leaves': 50,
              'min_child_samples': 20,
              'max_depth': 6,
              'colsample_bytree': 0.7,
              'verbosity': 5
              }
    lg=LGBMClassifier(silent=False,**params)
    lg.fit(X_train,y_train)

    #模型保存
    import pickle
    pickle.dump(lg,open("HappyBank_LightGBM_.pkl",'wb'))

    #输出特征重要性
    df=pd.DataFrame({'columns':list(feat_names),'importance':list(lg.feature_importances_.T)})
    df=df.sort_values(by='importance',ascending=False)
    print(df)

