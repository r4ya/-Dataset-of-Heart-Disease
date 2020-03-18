# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:34:37 2020

@author: ryavuz
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('C:\\Users\\ryavu\\Desktop\\heart-disease-uci\\heart.csv')
print(data.shape)
print(data.info())

print(data.isnull().sum())
print(data.nunique())

labels_cp = data.cp.value_counts().index
values_cp = data.cp.value_counts().values
print(data.cp.describe())
explode = [0.05,0.01,0.01,0.01]
plt.figure(figsize = (7,7))
plt.pie(values_cp, explode = explode, labels = labels_cp, autopct = '%1.1f%%')
plt.title("Chest Pain Types")
plt.savefig('pie-1.png')
plt.show()

pd.crosstab(data.cp,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for CP')
plt.xlabel('CP')
plt.ylabel('Frequency')
plt.savefig('bar-1.png')
plt.show()

labels_ca = data.ca.value_counts().index
values_ca = data.ca.value_counts().values
print(data.ca.describe())
explode = [0,0,0,0,0.1]
plt.figure(figsize = (7,7))
plt.pie(values_ca, explode = explode, labels = labels_ca, autopct = '%1.1f%%')
plt.title("number of major vessels (0-3) colored by flourosopy")
plt.savefig('pie2.png')
plt.show()

pd.crosstab(data.ca,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for CA')
plt.xlabel('CA')
plt.ylabel('Frequency')
plt.savefig('bar2.png')
plt.show()

labels_thal = data.thal.value_counts().index
values_thal = data.thal.value_counts().values
print(data.thal.describe())
explode = [0,0,0,0.1]
plt.figure(figsize = (7,7))
plt.pie(values_thal, explode = explode, labels = labels_thal, autopct = '%1.1f%%')
plt.title("thalach")
plt.savefig('pie3.png')
plt.show()

pd.crosstab(data.thal,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Thalach')
plt.xlabel('Thalach')
plt.ylabel('Frequency')
plt.savefig('bar3.png')
plt.show()

labels_exang = data.exang.value_counts().index
labels_exang = ['No','Yes']
values_exang = data.exang.value_counts().values
print(data.exang.describe())
explode = [0,0.1]
plt.figure(figsize = (7,7))
plt.pie(values_exang, explode = explode, labels = labels_exang, autopct = '%1.1f%%')
plt.title("Exercise Induced Angina")
plt.savefig('pie4.png')
plt.show()

pd.crosstab(data.exang,data.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Exercise Induced Angina')
plt.xlabel('Exercise Induced Angina')
plt.ylabel('Frequency')
plt.savefig('bar4.png')
plt.show()

labels_slope = data.slope.value_counts().index
values_slope = data.slope.value_counts().values
print(data.slope.describe())
explode = [0,0,0.1]
plt.figure(figsize = (7,7))
plt.pie(values_slope, explode = explode, labels = labels_slope, autopct = '%1.1f%%')
plt.title("the slope of the peak exercise ST segment")
plt.savefig('pie5.png')
plt.show()

pd.crosstab(data.slope,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('Slope')
plt.ylabel('Frequency')
plt.savefig('bar5.png')
plt.show()

labels_fbs = ['fbs < 120 mg/dl','fbs > 120 mg/dl']
values_fbs = data.fbs.value_counts().values
print(data.fbs.describe())
explode = [0,0.1]
plt.figure(figsize = (7,7))
plt.pie(values_fbs, explode = explode, labels = labels_fbs, autopct = '%1.1f%%')
plt.title("Fasting Blood Sugar")
plt.savefig('pie6.png')
plt.show()

pd.crosstab(data.fbs,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Fasting Blood Sugar')
plt.xlabel('Fasting Blood Sugar')
plt.ylabel('Frequency')
plt.savefig('bar6.png')
plt.show()

labels_restecg = data.restecg.value_counts().index
values_restecg = data.restecg.value_counts().values
print(data.restecg.describe())
explode = [0.02,0.02,0.05]
plt.figure(figsize = (7,7))
plt.pie(values_restecg, explode = explode, labels = labels_restecg, autopct = '%1.1f%%')
plt.title("Resting Electrocardiographic Results")
plt.savefig('pie7.png')
plt.show()

pd.crosstab(data.restecg,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Restecg')
plt.xlabel('Restecg')
plt.ylabel('Frequency')
plt.savefig('bar7.png')
plt.show()

pd.crosstab(data.age,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('bar8.png')
plt.show()

pd.crosstab(data.sex,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for SEX')
plt.xlabel('Sex (0 = Female, 1 = Male)' )
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.savefig('bar9.png')
plt.show()

pd.crosstab(data.trestbps,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for TrestBps')
plt.xlabel('TrestBps' )
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.savefig('bar10.png')
plt.show()

pd.crosstab(data.chol,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for CHOL')
plt.xlabel('Chol' )
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.savefig('bar11.png')
plt.show()

pd.crosstab(data.oldpeak,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for OldPeak')
plt.xlabel('OldPeak' )
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.savefig('bar12.png')
plt.show()

pd.crosstab(data.thalach,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Thalach')
plt.xlabel('Thalach')
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.savefig('bar13.png')
plt.show()
#%% H I S T O G R A M
data['AgeBin'] = 0 #creates a column of 0
data.loc[((data['age'] > 28) & (data['age'] < 30)) , 'AgeBin'] = 1
data.loc[((data['age'] >= 30) & (data['age'] < 40)) , 'AgeBin'] = 2
data.loc[((data['age'] >= 40) & (data['age'] < 50)) , 'AgeBin'] = 3
data.loc[((data['age'] >= 50) & (data['age'] < 60)) , 'AgeBin'] = 4
data.loc[((data['age'] >= 60) & (data['age'] < 70)) , 'AgeBin'] = 5
data.loc[((data['age'] >= 70) & (data['age'] < 78)) , 'AgeBin'] = 6
plt.figure()
plt.title('Age --- (29,77)')
data.AgeBin.hist()
plt.savefig('histogram.png')
plt.show()
#%% N O R M A L I Z A T I O N
def norm(data):
    return (data)/(max(data)-min(data))
#%% L I N E P L O T
norm(data.chol).plot(kind = 'line', color = 'r',label = 'Target',linewidth=1,alpha = 0.9,grid = True,linestyle = ':')
norm(data.trestbps).plot(color = 'g',label = 'Sex',linewidth=1, alpha = 0.4,grid = True,linestyle = '-.')
norm(data.thal).plot(color = 'b',label = 'Thalach',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     
plt.xlabel('x axis')            
plt.ylabel('y axis')
plt.title('Line Plot')
plt.savefig('line-plot-chol-trestbps-thal.png')
plt.show()
#%% S C A T T E R P L O T
data.plot(kind='scatter', x='chol', y='trestbps',alpha = 0.5,color = 'red')
plt.xlabel('Chol')          
plt.ylabel('Trestbps')
plt.title('Chol-TrestBps Scatter Plot')
plt.savefig('scatter-plot-chol-trestbps.png')
plt.show()

data.plot(kind='scatter', x='chol', y='thal',alpha = 0.5,color = 'blue')
plt.xlabel('Chol')         
plt.ylabel('Thalach')
plt.title('Chol-Thalach Scatter Plot')           
plt.savefig('scatter-plot-chol-thalach.png')
plt.show()
#%% P L O T and S U B P L O T
data1 = data.loc[:,["chol","trestbps","thalach"]]
data1.plot()
plt.savefig('plot.png')
plt.show()

data1.plot(subplots = True)
plt.savefig('subplot.png')
plt.show()
#%% B A R P L O T
plt.figure(figsize=(15,10))
sns.barplot(x=data.trestbps, y=data.thalach)
plt.xticks(rotation= 90)
plt.xlabel('TrestBps')
plt.ylabel('Thalach')
plt.title('TrestBps-Thalach')
plt.savefig('barplot11.png')
plt.show()

plt.figure(figsize=(15,10))
ax= sns.barplot(x=data.age, y=data.target)
plt.xlabel('Age')
plt.ylabel('Target')
plt.title('Target-Age Seaborn Bar Plot')
plt.savefig('barplot22.png')
plt.show()
#%% S U B P L O T
f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=data.sex,y=data.target,color='blue',alpha = 1.0,label='Sex')
sns.barplot(x=data.exang,y=data.target,color='red',alpha = 1.0,label='Exang')
sns.barplot(x=data.fbs,y=data.target,color='green',alpha = 0.5,label='Fbs')

ax.legend(loc='best',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Percentage of Features', ylabel='Target',title = "Features-Target")
plt.savefig('barplot-sex-exang-fbs.png')
plt.show()
#%% B O X P L O T
plt.figure(figsize=(20,10))
sns.boxplot(x="age", y="chol", hue="target", data=data, palette="PRGn")
plt.savefig('boxplot-age-chol.png')
plt.show()

plt.figure(figsize=(20,10))
sns.boxplot(x="trestbps", y="chol", hue="target", data=data, palette="PRGn")
plt.savefig('boxplot-trestbps-chol.png')
plt.show()

plt.figure(figsize=(20,10))
sns.boxplot(x="sex", y="cp", hue="target", data=data, palette="PRGn")
plt.savefig('boxplot-sex-cp.png')
plt.show()
#%% C O U N T P L O T
plt.figure(figsize=(10,7))
sns.barplot(x=data['age'].index,y=data['target'].values)
plt.savefig('barplot-age-target1.png')
plt.xlabel('Ages')

sns.countplot(x=data.age)
plt.ylabel('Number of Target People')
plt.title('Age of target people',color = 'blue',fontsize=15)
plt.savefig('countplot-age.png')
plt.show()
#%% H E A T M A P
features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang',
            'oldpeak','slope','ca','thal','target']
data = data[features]
f,ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.savefig('heatmap.png')
plt.show()
#%% J O I N T P L O T
sns.jointplot(data.cp, data.target, kind="kde", size=5, color ="g")
plt.savefig('kdeplot-cp.png')
plt.show()

sns.jointplot(data.thalach, data.target, kind="kde", size=5, color="r")
plt.savefig('kdeplot-thalach.png')
plt.show()

sns.jointplot(data.slope, data.target, kind="kde", size=5, color="b")
plt.savefig('kdeplot-slope.png')
plt.show()

sns.jointplot(data.restecg, data.target, kind="kde", size=5, color="cyan")
plt.savefig('kdeplot-restecg.png')
plt.show()

sns.kdeplot(data.age, data.target, shade=True, cut=3)
plt.savefig('kdeplot-age.png')
plt.show()
#%% L M P L O T 
sns.lmplot(x="slope", y="target", data=data)
plt.savefig('lmplot-slope.png')
plt.show()

sns.lmplot(x="cp", y="target",data=data)
plt.savefig('lmplot-cp.png')
plt.show()

sns.lmplot(x="cp", y="slope",hue="target",data=data,markers=["o", "x"])
plt.savefig('lmplot-cp-slope.png')
plt.show()

sns.lmplot(x="cp", y="thalach", hue="target", col="slope",
               data=data,markers=["o", "x"], height=6, aspect=.4, x_jitter=.1)
plt.savefig('lmplot-cp-thalach-slope.png')
plt.show()

sns.lmplot(x="cp", y="thalach", row="slope", col="restecg",hue="target",
               markers=["o", "x"],data=data, height=3)
plt.savefig('lmplot-cp-thalach-slope-restecg.png')
plt.show()
#%% P O I N T P L O T
ax = sns.pointplot(x="cp", y="thalach", hue="target",data=data,dodge=True)
plt.savefig('pointplot-cp-thalach.png')
plt.show()

ax = sns.pointplot(x="cp", y="slope", hue="target",data=data,dodge=True,
                   markers=["o", "x"],linestyles=["-", "--"])
plt.savefig('pointplot-cp-slope.png')
plt.show()

ax = sns.pointplot(x="slope", y="thalach", hue="target",data=data,dodge=True,
                   markers=["o", "x"],linestyles=["-", "--"])
plt.savefig('pointplot-slope-thalach.png')
plt.show()
#%% S W A R M P L O T
sns.swarmplot(x="cp", y="thalach",hue="target", data=data)
plt.savefig('swarmplot-cp-thalach.png')
plt.show()

sns.swarmplot(x="slope", y="thalach",hue="target", data=data)
plt.savefig('swarmplot-slope-thalach.png')
plt.show()

sns.swarmplot(x="exang", y="thalach",hue="target", data=data)
plt.savefig('swarmplot-exang-thalach.png')
plt.show()

sns.swarmplot(x="restecg", y="thalach",hue="target", data=data)
plt.savefig('swarmplot-restecg-thalach.png')
plt.show()

sns.swarmplot(x="thal", y="thalach",hue="target", data=data)
plt.savefig('swarmplot-thal-thalach.png')
plt.show()
#%% V I O L I N P L O T
plt.figure(figsize=(10,7))
sns.violinplot(x="cp", y="slope", hue="target",data=data, palette="muted")
plt.savefig('violinplot-cp-slope.png')
plt.show()

sns.violinplot(x="cp", y="slope", hue="target",data=data, palette="muted", split=True)
plt.savefig('violinplot-cp-slope.png')
plt.show()

sns.violinplot(x="cp", y="thalach", hue="target",data=data, palette="muted", split=True)
plt.savefig('violinplot-cp-thalach.png')
plt.show()

sns.violinplot(x="slope", y="thalach", hue="target",data=data, palette="muted", split=True)
plt.savefig('violinplot-slope-thalach.png')
plt.show()

sns.violinplot(x="exang", y="restecg", hue="target",data=data, palette="muted", split=True)
plt.savefig('violinplot-exang-restecg.png')
plt.show()

sns.violinplot(x="exang", y="thal", hue="target",data=data, palette="muted", split=True)
plt.savefig('violinplot-exang-thal.png')
plt.show()

sns.violinplot(x="restecg", y="thal", hue="target",data=data, palette="muted", split=True)
plt.savefig('violinplot-restecg-thal.png')
plt.show()
#%% A L G O R I T H M S
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score,roc_curve

features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang',
            'oldpeak','slope','ca','thal']
target = ['target']
x = data[features]
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
mlreg = LinearRegression()
mlreg.fit(x_train, y_train)
predictionmlreg = mlreg.predict(x_test)
test_set_rmse = (np.sqrt(mean_squared_error(y_test, predictionmlreg)))
print("Intercept: \n", mlreg.intercept_)
print("Root Mean Square Error \n", test_set_rmse)
rocaucscore=roc_auc_score(y_test.values, predictionmlreg)
print('Roc Score: ',rocaucscore)
#%%
dtclassifieroptimal = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4,
                       max_features=None, max_leaf_nodes=10,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=42, splitter='best')
dtclassifieroptimal.fit(x_train, y_train)
dtprediction = dtclassifieroptimal.predict(x_test)
print('Accuracy of Decision Tree:', accuracy_score(dtprediction,y_test))

from sklearn.metrics import confusion_matrix
cmdt=confusion_matrix(y_test, dtprediction, labels=None, sample_weight=None)
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cmdt, fmt = "d",
            xticklabels=['Have not Disease', 'Have Disease'],
            yticklabels=['Have not Disease', 'Have Disease'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Purple", cmap="Purples")
plt.title('Confusion Matrix in DT', fontsize=14)
plt.savefig('CMDT.png')
plt.show()
rocaucscore=roc_auc_score(y_test.values, dtprediction)
print('Roc Score: ',rocaucscore)
#%%
from sklearn.neighbors import KNeighborsClassifier
knnclassifier=KNeighborsClassifier(n_neighbors=6,algorithm='auto',
                                    leaf_size=30,metric='manhattan')
knnclassifier.fit(x_train, y_train)
trainaccuracy=knnclassifier.score(x_train, y_train)
testaccuracy=knnclassifier.score(x_test, y_test)
knnprediction=knnclassifier.predict(x_test)
print('train accuracy: {}\ntest accuracy: {}\n'.format(trainaccuracy,testaccuracy))

cmknn=confusion_matrix(y_test, knnprediction, labels=None, sample_weight=None)
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cmknn, fmt="d",
            xticklabels=['Have not Disease', 'Have Disease'],
            yticklabels=['Have not Disease', 'Have Disease'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Blue", cmap="Blues")
plt.title('Confusion Matrix in KNN', fontsize=14)
plt.savefig('CMKNN.png')
plt.show()
#%%
from sklearn.svm import SVC
svm = SVC(kernel='rbf',random_state = 42, tol=0.001, shrinking=True, probability=True,
          C=1.0,  degree=3, gamma='auto', coef0=0.0, cache_size=200, class_weight=None, 
          verbose=False, max_iter=-1)
svm.fit(x_train,y_train)
svmprediction = svm.predict(x_test)
print("print accuracy of svm algo: ",svm.score(x_test,y_test))

cmsvm=confusion_matrix(y_test, knnprediction, labels=None, sample_weight=None)
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cmknn, fmt="d",
            xticklabels=['Have not Disease', 'Have Disease'],
            yticklabels=['Have not Disease', 'Have Disease'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Green", cmap="Greens")
plt.title('Confusion Matrix in SVM', fontsize=14)
plt.savefig('CMSVM.png')
plt.show()
rocaucscore=roc_auc_score(y_test.values, svmprediction)
print(rocaucscore)
#%%
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
nbprediction = nb.predict(x_test)
print("print accuracy of naive bayes algo: ",nb.score(x_test,y_test))
cmnb=confusion_matrix(y_test, nbprediction, labels=None, sample_weight=None)
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cmnb, fmt="d",
            xticklabels=['Have not Disease', 'Have Disease'],
            yticklabels=['Have not Disease', 'Have Disease'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Orange", cmap="Oranges")
plt.title('Confusion Matrix in NaiveBayes', fontsize=14)
plt.savefig('CMNB.png')
plt.show()
rocaucscore=roc_auc_score(y_test.values, nbprediction)
print(rocaucscore)
#%%
from sklearn.ensemble import RandomForestClassifier
rfclassifieroptimal=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=4, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
rfclassifieroptimal.fit(x_train, y_train)
rfprediction = rfclassifieroptimal.predict(x_test)
print('Accuracy of Random Forest Classifier:', accuracy_score(rfprediction,y_test))
cmrf=confusion_matrix(y_test, rfprediction, labels=None, sample_weight=None)
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cmrf, fmt="d",
            xticklabels=['Have not Disease', 'Have Disease'],
            yticklabels=['Have not Disease', 'Have Disease'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Red", cmap="Reds")
plt.title('Confusion Matrix in Random Forest', fontsize=14)
plt.savefig('CMRF.png')
plt.show()
rocaucscore=roc_auc_score(y_test.values, rfprediction)
print(rocaucscore)
#%%
# calculate the fpr and tpr for all thresholds of the classification
y_pred_proba_DT = dtclassifieroptimal.predict_proba(x_test)[::,1]
fpr1, tpr1, _ = roc_curve(y_test,  y_pred_proba_DT)
auc1 = roc_auc_score(y_test, y_pred_proba_DT)

y_pred_proba_KNN = knnclassifier.predict_proba(x_test)[::,1]
fpr2, tpr2, _ = roc_curve(y_test,  y_pred_proba_KNN)
auc2 = roc_auc_score(y_test, y_pred_proba_KNN)

y_pred_proba_SVM = svm.predict_proba(x_test)[::,1]
fpr3, tpr3, _ = roc_curve(y_test,  y_pred_proba_SVM)
auc3 = roc_auc_score(y_test, y_pred_proba_SVM)

y_pred_proba_NB = nb.predict_proba(x_test)[::,1]
fpr4, tpr4, _ = roc_curve(y_test,  y_pred_proba_NB)
auc4 = roc_auc_score(y_test, y_pred_proba_NB)

y_pred_proba_RF = svm.predict_proba(x_test)[::,1]
fpr5, tpr5, _ = roc_curve(y_test,  y_pred_proba_RF)
auc5 = roc_auc_score(y_test, y_pred_proba_RF)

plt.figure(figsize=(10,7))
plt.title('ROC', size=15)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr1,tpr1,label="DT, auc="+str(round(auc1,2)))
plt.plot(fpr2,tpr2,label="KNearest Neighbor, auc="+str(round(auc2,2)))
plt.plot(fpr3,tpr3,label="SVM, auc="+str(round(auc3,2)))
plt.plot(fpr4,tpr4,label="NB, auc="+str(round(auc4,2)))
plt.plot(fpr5,tpr5,label="RF, auc="+str(round(auc5,2)))
plt.legend(loc='best', title='Models', facecolor='white')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.box(False)
plt.savefig('ROCAUCCURVE.png')
plt.show()
#%%
plt.figure(figsize=(24,4))
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1,5,1)
plt.title("Decision Tree Confusion Matrix")
sns.heatmap(cmdt,annot=True,cmap="Purples",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(1,5,2)
plt.title("K-Nearest Neighbors Confusion Matrix")
sns.heatmap(cmknn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(1,5,3)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cmsvm,annot=True,cmap="Greens",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(1,5,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cmnb,annot=True,cmap="Oranges",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(1,5,5)
plt.title("Random Forest Classifier Confusion Matrix")
sns.heatmap(cmrf,annot=True,cmap="Reds",fmt="d",cbar=False, annot_kws={"size": 24})

plt.savefig('Comparison-of-confusion-matrix.png')
plt.show()