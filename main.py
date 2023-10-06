import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn 
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import time

startTimer = time.time()
# Referenced from adult.names
columns = ["age", "workclass", "fnlwgt", "education", "education-num","marital-status", "occupation", "relationship",
          "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

train = pd.read_csv('data/adult.data', names=columns, sep=' *, *', engine='python', na_values='?')
test = pd.read_csv('data/adult.test', names=columns, sep=' *, *', engine='python', skiprows=1, na_values='?')
train.head()
test.head()

train[train=='?']=np.nan
train = train.dropna(axis=0)
test[test=='?']=np.nan
test = test.dropna(axis=0)
cat_attributes = train.select_dtypes(include=['object'])
cat_attributes.describe()
num_attributes = train.select_dtypes(include=['int'])
num_attributes.describe()
train["income"]=train["income"].map({"<=50K":0,">50K":1})
test["income"]=test["income"].map({"<=50K.":0,">50K.":1})

a=['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','income']
for i in a:
    print(i,':',stats.pointbiserialr(train['income'],train[i])[0])

for col in train.columns:
    if train[col].dtypes == 'object':
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        
for col in test.columns:
    if test[col].dtypes == 'object':
        le = LabelEncoder()
        test[col] = le.fit_transform(test[col].astype(str))


numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

count_incoding_list = dict(train['native-country'].value_counts())
train['native-country'] = train['native-country'].map(count_incoding_list)
train[['native-country']] = MinMaxScaler().fit_transform(train[['native-country']])
train[numerical] = MinMaxScaler().fit_transform(train[numerical])

count_incoding_list = dict(test['native-country'].value_counts())
test['native-country'] = test['native-country'].map(count_incoding_list)
test[['native-country']] = MinMaxScaler().fit_transform(test[['native-country']])
test[numerical] = MinMaxScaler().fit_transform(test[numerical])


train_X=train.drop(["fnlwgt","education"],axis=1)
test_X=test.drop(["fnlwgt","education"],axis=1)

train_y=train["income"]
test_y=test["income"]

del train_X["income"]
del test_X["income"]

def plot(column):
    if train[column].dtype != 'int64':
        f, axes = plt.subplots(1,1,figsize=(15,5))
        sns.countplot(x=column, hue='income', data = train)
        plt.xticks(rotation=90)
        plt.suptitle(column,fontsize=20)
        plt.show()
    else:
        g = sns.FacetGrid(train, row="income", margin_titles=True, aspect=4, height=3)
        g.map(plt.hist,column,bins=100)
        plt.show()
    plt.show()

def plotKnn():
    error=[]
    for i in range(15,35,1):
        knn_model = KNeighborsClassifier(n_neighbors=i)
        knn_model.fit(train_X,train_y)
        knn_pred = knn_model.predict(test_X)
        error.append(np.mean(knn_pred != test_y))

        print('\nk = ', i )
        print('Train Accuracy Score = ', round(knn_model.score(train_X, train_y) * 100, 4),"%")
        print("Test Accuracy Score:", round(accuracy_score(test_y, knn_pred) * 100, 4),"%")
        print("F1 Score: ", round(f1_score(test_y, knn_pred) * 100,4),"%")
        print("MSE: ", round(mean_squared_error(test_y, knn_pred) * 100,4),"%")
        
        print(confusion_matrix(test_y,knn_pred))
        print(classification_report(test_y,knn_pred))

    plt.figure(figsize=(12, 6))
    plt.plot(range(15,35,1), error, color='red', linestyle='dashed', marker='x',markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Squared Error')
    plt.show()

def predictKnn():
    startTimer = time.time()
    kvalue = 0
    compareValue = 100
    error=[]
    for i in range(15,35,1):
        knn_model = KNeighborsClassifier(n_neighbors=i)
        knn_model.fit(train_X,train_y)
        knn_pred = knn_model.predict(test_X)
        error.append(np.mean(knn_pred != test_y))
        if(round(mean_squared_error(test_y, knn_pred) * 100,4)<compareValue): 
            compareValue=round(mean_squared_error(test_y, knn_pred) * 100,4) 
            kvalue = i
    metric_accuracy_training = {}
    metric_accuracy_testing = {}

    dist_calc = ['euclidean', 'manhattan']

    for i in dist_calc:
        model = KNeighborsClassifier(kvalue, metric = i) 
        model.fit(train_X, train_y)
        metric_accuracy_training[i] = model.score(train_X, train_y)
        metric_accuracy_testing[i] = model.score(test_X, test_y) 
        stopTimer = time.time()
        print('\nCalculation : ', i)
        print('Training accuracy :  ', knn_model.score(train_X, train_y))
        print('Testing accuracy : ', knn_model.score(test_X, test_y))
        print('Computational time :', time.time() - startTimer)

predictKnn()

