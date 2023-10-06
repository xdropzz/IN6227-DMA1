import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn 
import scipy.stats as stats
import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_val_predict


# Pre-processing data
startTimer = time.time()
kvalue = 0
# Referenced from adult.names
columns = ["age", "workclass", "fnlwgt", "education", "education-num","marital-status", "occupation", "relationship",
          "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

train = pd.read_csv('data/adult.data', names=columns, sep=' *, *', engine='python', na_values='?')
train[train=='?']=np.nan
train = train.dropna(axis=0)
test = pd.read_csv('data/adult.test', names=columns, sep=' *, *', engine='python', skiprows=1, na_values='?')
test[test=='?']=np.nan
test = test.dropna(axis=0)

cat_attributes = train.select_dtypes(include=['object'])
# cat_attributes.describe()
num_attributes = train.select_dtypes(include=['int'])
# num_attributes.describe()
train["income"]=train["income"].map({"<=50K":0,">50K":1})
test["income"]=test["income"].map({"<=50K.":0,">50K.":1})

# Examining Biserial correlation
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


train_x=train.drop(["fnlwgt","education"],axis=1)
test_x=test.drop(["fnlwgt","education"],axis=1)

train_y=train["income"]
test_y=test["income"]

del train_x["income"]
del test_x["income"]

print('\n----------- STARTING DATA PRE-PROCESSING -----------')
print('Time taken to pre-process data', time.time() - startTimer)
print('----------- END OF DATA PRE-PROCESSING -----------')

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

def predictKnn():
    print('\n----------- FIND BEST KNN COMPUTATION -----------')
    startTimer = time.time()
    compareValue = 100
    error=[]
    for i in range(15,36,1):
        knn_model = KNeighborsClassifier(n_neighbors=i)
        knn_model.fit(train_x,train_y)
        knn_pred = knn_model.predict(test_x)
        error.append(np.mean(knn_pred != test_y))
        if(round(mean_squared_error(test_y, knn_pred) * 100,4)<compareValue): 
            compareValue=round(mean_squared_error(test_y, knn_pred) * 100,4) 
            kvalue = i
    stopTimer = time.time()
    knn_model = KNeighborsClassifier(n_neighbors=kvalue)
    knn_model.fit(train_x,train_y)
    knn_pred = knn_model.predict(test_x)
    print('Train Accuracy Score = ', round(knn_model.score(train_x, train_y) * 100, 2),"%")
    print("Test Accuracy Score:", round(accuracy_score(test_y, knn_pred) * 100, 2),"%")
    print("F1 Score: ", round(f1_score(test_y, knn_pred) * 100,2),"%")
    print("MSE: ", round(mean_squared_error(test_y, knn_pred) * 100,2),"%")
    print(confusion_matrix(test_y,knn_pred))
    print(classification_report(test_y,knn_pred))
    print('best kvalue =', kvalue,'| compute time: ',stopTimer-startTimer)
    print('\n----------- START KNN COMPUTATION (Euclidean & Manhattan) -----------')
    startTimer = time.time()
    metric_accuracy_training = {}
    metric_accuracy_testing = {}
    dist_calc = ['euclidean', 'manhattan']
    for i in dist_calc:
        model = KNeighborsClassifier(kvalue, metric = i) 
        model.fit(train_x, train_y)
        metric_accuracy_training[i] = model.score(train_x, train_y)
        metric_accuracy_testing[i] = model.score(test_x, test_y) 
        stopTimer = time.time()
        print('\nCalculation : ', i)
        print('Training Accuracy :  ', round(knn_model.score(train_x, train_y)*100, 2),'%')
        print('Testing Accuracy : ', round(knn_model.score(test_x, test_y)*100, 2),'%')
        print('Computational Time :', stopTimer - startTimer)
    print('\nTotal Compute Time:', time.time()-startTimer)
    print('----------- END KNN COMPUTATION -----------')


predictKnn()