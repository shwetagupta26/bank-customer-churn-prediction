#!/usr/bin/env python
# coding: utf-8

# Loading the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


#read and store the dataset from csv file
df = pd.read_csv("C:\\Users\shwet\Documents\shweta\my_project\Churn_Modelling.csv")
df.head(10)


# In[3]:


#check missing values
df.isnull().sum()


# In[4]:


#dropping the below coloumns as they as the are specific to a customer.
df= df.drop(columns = ['RowNumber','CustomerId','Surname'])


# Performing exploratory data analysis

# In[5]:


exitcount = df.Exited[df.Exited==1].count()
nonexitcount = df.Exited[df.Exited==0].count()
labels = 'Exited','Retained'
sizes = [exitcount, nonexitcount]
explode = (0.1,0)
plt.pie(sizes, explode=explode, labels=labels,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Proportion of customer churned and retained',size=20)
plt.show()


# In[6]:


fig,axarr = plt.subplots(2, 2, figsize=(15, 8))
sns.countplot(df.Geography, hue=df.Exited,ax= axarr[0,0])
sns.countplot(df.Gender, hue=df.Exited,ax=axarr[0,1])
sns.countplot(df.HasCrCard, hue=df.Exited,ax=axarr[1,0])
sns.countplot(df.IsActiveMember, hue=df.Exited,ax=axarr[1,1])


# In[7]:


fig,axarr = plt.subplots(3, 2, figsize=(18, 12))
sns.boxplot(df.Exited,df.Age,hue =df.Exited,ax= axarr[0,0])
sns.boxplot(df.Exited,df.CreditScore,hue =df.Exited,ax= axarr[0,1])
sns.boxplot(df.Exited,df.Tenure,hue =df.Exited,ax= axarr[1,0])
sns.boxplot(df.Exited,df.Balance,hue =df.Exited,ax= axarr[1,1])
sns.boxplot(df.Exited,df.NumOfProducts,hue =df.Exited,ax= axarr[2,0])
sns.boxplot(df.Exited,df.EstimatedSalary,hue =df.Exited,ax= axarr[2,1])


# In[8]:


pd.crosstab(df.Gender,df.Exited).plot(kind='bar')


# In[9]:


# introducing new variables to standardize the variables.
df['BalanceSalaryRatio'] = df.Balance/df.EstimatedSalary
df['TenureByAge'] = df.Tenure/(df.Age)
df['CreditScoreGivenAge'] = df.CreditScore/(df.Age)
df.head(10)


# In[10]:


# minMax scaling the continuous variables
continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember','Geography', 'Gender']
df[continuous_vars].min().copy()
minVec = df[continuous_vars].min().copy()
maxVec = df[continuous_vars].max().copy()
df[continuous_vars] = (df[continuous_vars]-minVec)/(maxVec-minVec)
df.head(10)


# In[11]:


# on ehot encoding for the variable Geography,Gender
df_new= pd.get_dummies( df, columns = ['Geography','Gender'] )
df_new.head(10)


# Feature engineering

# In[12]:


# Split Train, test data
df_train = df_new.sample(frac=0.8,random_state=200)
df_test = df_new.drop(df_train.index)

# Another method to split the data into train and test
y= df_new.Exited
x= df_new.drop('Exited',axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train.head(10)


# In[13]:


# Fit models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier

# Scoring functions
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[14]:


#building logsitic regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))


# In[33]:


from sklearn.metrics import classification_report
print('Logsitic Regression Model Accuracy: ')
print(classification_report(y_test, y_pred))


# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier to not label a sample as positive if it is negative.
# 
# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
# 
# The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.

# In[16]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[30]:


logit_roc_auc = roc_auc_score(y_test, y_pred)
print('ROC Score'+str(logit_roc_auc))
fpr_logreg, tpr_log_reg, thresholds_log_reg = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr_logreg, tpr_log_reg, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[28]:


#building decision tree
from sklearn.tree import DecisionTreeClassifier
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
# Make a decision tree and train
dtree = DecisionTreeClassifier()
dtree.fit(x_train,y_train)
y_pred = dtree.predict(x_test)
print('Model Accuracy: {decision Tree score(x_test,y_test)}')
print(classification_report(y_test, y_pred))

dt_roc_auc = roc_auc_score(y_test, y_pred)
print('ROC Score:'+ str(dt_roc_auc))
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, dtree.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr_dt, tpr_dt, label='Logistic Regression (area = %0.2f)' % dt_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[43]:


#building confusion Matrix
from sklearn.metrics import confusion_matrix
import itertools
confusion_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrixb
plt.figure(figsize = (8, 5))
cmap = plt.cm.Oranges
classes = ['Exited', 'Retained']
plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
plt.title('Decision Tree Confusion matrix', size = 18)
plt.colorbar(aspect=4)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45, size = 14)
plt.yticks(tick_marks, classes, size = 14)

fmt = 'd'
thresh = confusion_matrix.max() / 2.
for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
    plt.text(j, i, format(confusion_matrix[i, j], fmt), fontsize = 15,
             horizontalalignment="center",
             color="white" if confusion_matrix[i, j] > thresh else "black")

plt.grid(None)
plt.tight_layout()
plt.ylabel('True label', size = 18)
plt.xlabel('Predicted label', size = 18)


# In[46]:


rfmodel = RandomForestClassifier(n_estimators=100,
                               bootstrap = True)
rfmodel.fit(x_train,y_train)
y_pred = rfmodel.predict(x_test)
print('Random Forest Model Accuracy')
print(classification_report(y_test, y_pred))

rf_roc_auc = roc_auc_score(y_test, y_pred)
print(rf_roc_auc)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rfmodel.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr_rf, tpr_rf, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[54]:


from sklearn.ensemble import GradientBoostingClassifier

gbmodel = GradientBoostingClassifier(n_estimators=300)
gbmodel.fit(x_train,y_train)
y_pred = gbmodel.predict(x_test)
print(f'Model Accuracy: {gbmodel.score(x_test,y_test)}')
print(classification_report(y_test, y_pred))

gb_roc_auc = roc_auc_score(y_test, y_pred)
print(gb_roc_auc)
fpr_gb, tpr_gb, thresholds = roc_curve(y_test, gbmodel.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr_gb, tpr_gb, label='Random Forest (area = %0.2f)' % gb_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[48]:


# summarizing the ROC curve for all the models
plt.figure(figsize = (8,5), linewidth= 1)
plt.plot(fpr_logreg, tpr_log_reg, label = 'Logistic Regression Score: ' + str(round(logit_roc_auc, 5)))
plt.plot(fpr_dt, tpr_dt, label = 'Decision Tree Score: ' + str(round(dt_roc_auc, 5)))
plt.plot(fpr_rf, tpr_rf, label = 'RF score: ' + str(round(rf_roc_auc, 5)))
plt.plot(fpr_gb, tpr_gb, label = 'XGB score: ' + str(round(gb_roc_auc, 5)))
plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.5')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
#plt.savefig('roc_results_ratios.png')
plt.show()

