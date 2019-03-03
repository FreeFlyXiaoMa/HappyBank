import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
test=pd.read_csv('Test.csv',encoding='ISO-8859-1')
train=pd.read_csv('Train.csv',encoding='ISO-8859-1')
#print('Test',test.columns)
#print('Train',train.columns)
#print(train.describe())
#sn.countplot(train['Disbursed'])
#plt.xlabel('Disbursed')
#plt.ylabel('number of occurrence')
#plt.show()
#将测试数据与训练数据合并为一列，方便做特征工程，source列包括（trian,test)两种
train['source']='train'
test['source']='test'
data=pd.concat([train,test],ignore_index=False)
#统计为null数量
#print(data.apply(lambda s:sum(s.isnull())))
#print(data.columns)
#print(data.shape)
#data.to_csv('data.csv')
#为'City', 'Employer_Name', 'Salary_Account', 'Source'这四个字段分别设置阈值，小于这些阈值的为稀有值
cat_features = ['City', 'Employer_Name', 'Salary_Account', 'Source']
rare_thresholds = [100, 30, 40, 40]
j = 0
for col in cat_features:
    # 每个取值的样本数目
    value_counts_col = data[col].value_counts(dropna=False)

    # 样本数目小于阈值的取值为稀有值
    rare_threshold = rare_thresholds[j]
    value_counts_rare = list(value_counts_col[value_counts_col < rare_threshold].index)
    #value_counts_rare = list(value_counts_col[value_counts_col].index)

    # 稀有值合并为：others
    rare_index = data[col].isin(value_counts_rare)
    data.loc[data[col].isin(value_counts_rare), col] = "Others"

    j = j + 1

#创建一个年龄的字段Age
data['Age'] = pd.to_datetime(data['Lead_Creation_Date']).dt.year - pd.to_datetime(data['DOB']).dt.year
#data['Age'].head()

#把原始的DOB字段去掉:
data.drop(['DOB', 'Lead_Creation_Date'],axis=1,inplace=True)

#不合理的贷款年限，设为缺失值
data['Loan_Tenure_Applied'].replace([10,6,7,8,9],value = np.nan, inplace = True)
data['Loan_Tenure_Submitted'].replace(6, np.nan, inplace = True)

#不能用于预测特征，drop
data.drop('LoggedIn',axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
feats_to_encode = ['City', 'Employer_Name', 'Salary_Account','Device_Type','Filled_Form','Gender','Mobile_Verified','Source','Var1','Var2','Var4']
for col in feats_to_encode:
    data[col] = le.fit_transform(data[col].astype(str))

#print(data.head())
#将data中source为train的数据赋给train，test一样
train=data.loc[data['source']=='train']
test=data.loc[data['source']=='test']
train.drop(['source'],axis=1,inplace=True)
test.drop(['source','Disbursed'],axis=1,inplace=True)
#print('train.head()',train.head())
train.to_csv('Bank_FE_rain.csv',index=False)
test.to_csv('BankFE_test.csv',index=False)