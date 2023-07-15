import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

data=pd.read_csv('survey.csv')
data.head()

if data.isnull().sum().sum()==0 :
    print ('There is no missing data in our dataset')
else:
    print('There is {} missing data in our dataset'.format(data.isnull().sum().sum()))


frame=pd.concat([data.isnull().sum(), data.nunique(), data.dtypes],axis=1,sort=False)
frame

data['work_interfere'].unique()

ax=sns.countplot(data=data,x='work_interfere')
ax.bar_label(ax.containers[0])

data=data.drop(columns=['state','comments','Timestamp'])
data['work_interfere']=SimpleImputer(strategy='most_frequent').fit_transform(data['work_interfere'].values.reshape(-1,1))
data['self_employed']=SimpleImputer(strategy='most_frequent').fit_transform(data['self_employed'].values.reshape(-1,1))
data.head()

ax=sns.countplot(data=data,x='work_interfere')
ax.bar_label(ax.containers[0])

print(data['Gender'].unique())
print('')
print('-'*75)
print('')
print('number of unique Gender in our dataset is :',data['Gender'].nunique())

data['Gender'].replace(['Male','male','M','m','Male','Cis Male',
                        'Man','cis male','Mail','Male-ish','Male (CIS)',
                        'Cis Man','msle','Malr','Mal','maile','Make',],'Male',inplace=True)

data['Gender'].replace(['Female','female','F','f','Woman','Female',
                        'femail','Cis Female','cis-female/femme','Femake','Female (cis)',
                        'woman',],'Female',inplace=True)

data["Gender"].replace(['Female (trans)','queer/she/they','non-binary',
                        'fluid','queer','Androgyne','Trans-female','male leaning androgynous',
                        'Agender','A little about you','Nah','All',
                        'ostensibly male,unsure what that really means',
                        'Genderqueer','Enby','p','Neuter','something kinda male?',
                        'Guy (-ish) ^_^','Trans woman',],'Other',inplace=True)
print(data['Gender'].unique())

ax=sns.countplot(data=data,x='Gender')
ax.bar_label(ax.containers[0])

if data.isnull().sum().sum()==0:
    print('There is no missing data')
else:
    print('There is {} missing data'.format(data.isnull().sum().sum()))


if data.duplicated().sum()==0:
    print('There is no duplicated data:')
else:
    print('Tehre is {} duplicated data:'.format(data.duplicated().sum()))
    data.drop_duplicates(inplace=True)
print(data.duplicated().sum())

data['Age'].unique()

data.drop(data[data['Age']<0].index, inplace = True)
data.drop(data[data['Age']>99].index, inplace = True)
data['Age'].unique()

plt.figure(figsize=(10,6))
age_range_plot=sns.countplot(data=data,x='Age')
age_range_plot.bar_label(age_range_plot.containers[0])
plt.xticks(rotation=90)

plt.figure(figsize=(10,6))
sns.displot(data['Age'],kde='treatment')
plt.title('Distribution treatment by age')

plt.figure(figsize=(10,6))
treat=sns.countplot(data=data,x='treatment')
treat.bar_label(treat.containers[0])
plt.title('Total number of individuals who received treatment or not')

data.info()

le=LabelEncoder()
columns_to_encode=['Gender','Country','self_employed','family_history','treatment','work_interfere','no_employees',
                             'remote_work','tech_company','benefits','care_options','wellness_program',
                             'seek_help','anonymity','leave','mental_health_consequence','phys_health_consequence',
                             'coworkers','supervisor','mental_health_interview','phys_health_interview',
                             'mental_vs_physical','obs_consequence']
for columns in columns_to_encode:
    data[columns] = le.fit_transform(data[columns])
data.info()

data.describe()

data['Age']=MaxAbsScaler().fit_transform(data[['Age']])
data['Country']=StandardScaler().fit_transform(data[['Country']])
data['work_interfere']=StandardScaler().fit_transform(data[['work_interfere']])
data['no_employees']=StandardScaler().fit_transform(data[['no_employees']])
data['leave']=StandardScaler().fit_transform(data[['leave']])
data.describe()

x=data.drop(columns=['treatment'])
y=data['treatment']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

#RANDOM FOREST CLASSIFIER
steps_rfc = [('Scaler', StandardScaler()),('clf', RFC(n_estimators = 40))]
clf_rfc = Pipeline(steps=steps_rfc)
clf_rfc.fit(x_train, y_train)
y_pred_rfc = clf_rfc.predict(x_test)

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_rfc)*100
mse = mean_squared_error(y_true=y_test,y_pred=y_pred_rfc)
r2 = r2_score(y_test,y_pred_rfc)

print(f'Accuray:{accuracy}')
print(f'MSE:{mse}')
print(f'R2 Score:{r2}')