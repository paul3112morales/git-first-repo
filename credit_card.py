# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:40:16 2024

@author: ALONSO ESPEJO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix,classification_report

base_application = pd.read_csv("C:/Users/ALONSO ESPEJO/Desktop/Mmmmorales/Credit card/application_record.csv/application_record.csv")
base_credit_record = pd.read_csv("C:/Users/ALONSO ESPEJO/Desktop/Mmmmorales/Credit card/credit_record.csv/credit_record.csv")

#EDA

base_application.describe().T
base_application.info()

# Display data types of columns
print(base_application.dtypes)

base_application["CODE_GENDER"].value_counts()

#Clients in application
print(base_application['ID'].nunique())
print(base_credit_record['ID'].nunique())

print(base_application.info())

#Duplicates IDs
base_application['ID'].duplicated().sum() ## 47
base_application = base_application.drop_duplicates(subset='ID',keep='first')

#Null information
base_application.isnull().sum() ### Ocu´pation_type has nulls-
base_application['OCCUPATION_TYPE'].fillna('not_specified',inplace=True)

# Plotting the histogram

plt.hist(base_credit_record['MONTHS_BALANCE'], 
         bins=range(min(base_credit_record['MONTHS_BALANCE']), 
                    max(base_credit_record['MONTHS_BALANCE']) + 1), 
         edgecolor='black')  # Adjust the bin size as needed

plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.grid(True)
plt.show()

### Univariate analysis

#Numeric variables application
sns.boxplot(data=base_application, y=base_application["CNT_CHILDREN"])
plt.show()
sns.boxplot(data=base_application, y=base_application["AMT_INCOME_TOTAL"])
plt.show()
sns.boxplot(data=base_application, y=base_application["DAYS_BIRTH"])
plt.show()
sns.boxplot(data=base_application, y=base_application["DAYS_EMPLOYED"])
plt.show() # Outliers
sns.boxplot(data=base_application, y=base_application["CNT_FAM_MEMBERS"])
plt.show()

#Replacing value of 365243 to 0
base_application['DAYS_EMPLOYED'].replace(365243,0,inplace=True)

#Categorical variables application
print(base_application['CODE_GENDER'].value_counts())
print(base_application['FLAG_OWN_CAR'].value_counts())
print(base_application['FLAG_OWN_REALTY'].value_counts())
print(base_application['NAME_INCOME_TYPE'].value_counts())
print(base_application['NAME_EDUCATION_TYPE'].value_counts())
print(base_application['NAME_FAMILY_STATUS'].value_counts())
print(base_application['NAME_HOUSING_TYPE'].value_counts())

print(base_application['FLAG_MOBIL'].value_counts())
print(base_application['FLAG_WORK_PHONE'].value_counts())
print(base_application['FLAG_PHONE'].value_counts())
print(base_application['FLAG_EMAIL'].value_counts())
print(base_application['OCCUPATION_TYPE'].value_counts())

#Categorical variables credit record
print(base_credit_record['STATUS'].value_counts())

### Creating new columns
base_application['YEARS_WORKING'] = round(-base_application['DAYS_EMPLOYED']/365.2)
base_application.loc[base_application['YEARS_WORKING']<0,'YEARS_WORKING']=0

base_application['AGE']=round(-base_application['DAYS_BIRTH']/365.2,0)

base_application.drop(columns=["DAYS_BIRTH","DAYS_EMPLOYED", "FLAG_MOBIL"],inplace=True)

base_application.describe(percentiles=[.01,.02,.03,.04,.05,.1,.25,.5,.75,.9,.95,.96,.97,.98,.99]).T

cols_to_scale = ["CNT_CHILDREN", "AMT_INCOME_TOTAL", "AGE", "YEARS_WORKING", "CNT_FAM_MEMBERS"]
cols_to_ohe = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
               "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
               "FLAG_PHONE", "FLAG_WORK_PHONE", "FLAG_EMAIL", "OCCUPATION_TYPE"]

#Scaling numeric variables
scaled = StandardScaler()
base_application[cols_to_scale] = scaled.fit_transform(base_application[cols_to_scale])

#OHE
base_application = pd.get_dummies(base_application, columns = cols_to_ohe, dtype = int)

# Target definition
base_credit_record['target'] = base_credit_record['STATUS']
base_credit_record['target'].replace('X', 0, inplace=True)
base_credit_record['target'].replace('C', 0, inplace=True)
base_credit_record['target']=base_credit_record['target'].astype(int)
base_credit_record.loc[base_credit_record['target']>=1,'target']=1

# Getting max target per ID
base_target = pd.DataFrame(base_credit_record.groupby(['ID'])['target'].agg("max")).reset_index()
base_target["target"].value_counts()

base_modelar = pd.merge(base_application, base_target, how = "inner", on=['ID'])



## Months account were open
start_month=pd.DataFrame(base_credit_record.groupby(['ID'])['MONTHS_BALANCE'].agg(min)).reset_index()
start_month.rename(columns={'MONTHS_BALANCE':'ACCOUNT_LENGTH'}, inplace=True)
start_month['ACCOUNT_LENGTH']=-start_month['ACCOUNT_LENGTH']

base_modelar = pd.merge(base_modelar, start_month, how='inner', on=['ID'])

#Deleting ID
base_modelar.drop(columns=["ID"],inplace=True)

#Separating predictors and target
x = base_modelar.drop(columns = ['target'])
y = base_modelar['target']

#To prevent bias, is necessary to stratify y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y,
                                                    random_state=2024)

###Try importance feature

###### Modelling

###Logistic regression
lr = LogisticRegression(random_state=2024)
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

###Decesion tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=2024)
dt.fit(x_train,y_train)
dt.score(x_train,y_train),dt.score(x_test,y_test) # getting the accuracy
y_pred = lr.predict(x_test)

#Decision tree with grid search
params_gs={
    "criterion":["gini", "entropy"],
"max_depth": [5,6,7,9,10,11, 12],
"min_samples_split" :[10,15,20,50,100,200,250,500],
"min_samples_leaf" : [5,10,15,20,50,80,100,150]}

from sklearn.model_selection import RandomizedSearchCV
rs_dt = RandomizedSearchCV(DecisionTreeClassifier(random_state=0),param_distributions=params_gs,cv=5,n_jobs=2)
rs_dt.fit(x_train,y_train)
rs_dt.score(x_train,y_train),rs_dt.score(x_test,y_test) #more stable

y_pred = rs_dt.predict(x_test)
print(confusion_matrix(y_test,y_pred))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0)
rf.fit(x_train,y_train)
rf.score(x_train,y_train),rf.score(x_test,y_test)

params_rf={"criterion":['gini', 'entropy'],
           "max_depth":[5,6,7,8,9,11,10,12,13],
           "min_samples_split":[2,5,8,9,10,15,18,20],
           "min_samples_leaf":[1,2,5,8,10],
           "n_estimators":[50,100,150,200,250,300],
           "bootstrap":[True],
           "max_features":["sqrt","log2"],
           "max_samples":[.5,.6,.75,.8,.9]}

rs_rf = RandomizedSearchCV(RandomForestClassifier(random_state=2024),param_distributions=params_rf,cv=5,n_jobs=2)
rs_rf.fit(x_train,y_train)
rs_rf.score(x_train,y_train),rs_rf.score(x_test,y_test)
y_pred = rs_rf.predict(x_test)
print(confusion_matrix(y_test,y_pred))

#Importance variable
pd.DataFrame(rf.feature_importances_,rf.feature_names_in_) ### Account_length,AMT_INCOME_TOTAL,AGE


#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=2024)

gb.fit(x_train,y_train)
gb.score(x_train,y_train),gb.score(x_test,y_test)
y_pred = gb.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

## hyperparameter tuning
params_gb = {
    'max_depth': [2,3,5,8,10],
    'min_samples_leaf': [2,5,10,20,35,50],
    'n_estimators': [10,15,20,25,30],
    'learning_rate': [0.1,0.15,0.2,0.25,0.3,0.4,0.5]
}

rs_gb = RandomizedSearchCV(GradientBoostingClassifier(random_state=2024),param_distributions=params_gb,cv=5,n_jobs=2)
rs_gb.fit(x_train,y_train)

y_pred = rs_gb.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

rs_gb.best_estimator_


#ADABoost
from sklearn.ensemble import AdaBoostClassifier
model_AdaBoost=AdaBoostClassifier(base_estimator=
                                  DecisionTreeClassifier(random_state=2024),
                                  n_estimators=80,
                              random_state=2024)

#fit the model on the data and predict the values
model_AdaBoost.fit(x_train,y_train)
model_AdaBoost.score(x_train,y_train),model_AdaBoost.score(x_test,y_test)

y_pred = model_AdaBoost.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


#XGBoost
import xgboost as xgb
model_xgb = xgb.XGBClassifier(objective='binary:logistic',
                            eval_metric="logloss", ## this avoids a warning...
                            seed=2024,
                            use_label_encoder=False)

model_xgb.fit(x_train, y_train)
model_xgb.score(x_train,y_train),model_xgb.score(x_test,y_test)

y_pred = model_xgb.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

param_grid = {'max_depth': [3, 4, 5],'learning_rate': [0.1, 0.01, 0.05],
              'gamma': [0, 0.25, 1.0],'reg_lambda': [0, 1.0, 10.0],
              'scale_pos_weight': [1, 3, 5]}

optimal_params = RandomizedSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic',seed=0,
                                                          subsample=0.9,colsample_bytree=0.5,early_stopping_rounds=10, eval_metric='auc',
                                                          use_label_encoder=False),param_distributions=param_grid,
                              scoring='roc_auc', verbose=0,cv = 3)#multi:softmax in multiclass problems

optimal_params.fit(x_train,y_train,eval_set=[(x_test, y_test)])

y_pred = optimal_params.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#New classifiers are upcoming