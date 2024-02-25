#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

df = pd.read_csv("D:/MCA 3 SEM/BIG DATA LAB/2270325_HeartFailure.csv")
df


# In[5]:


# data exploration

df.columns


# In[6]:


df.shape


# In[7]:


df.isnull()


# In[8]:


mean_age = df["age"].mean()
print(mean_age)


# In[9]:


mean_deaths = df["deaths"].mean()
print(mean_deaths)


# In[10]:


#  importing the libraries

import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")


# Analysing the deaths and alives

labels = ["Healthy", "Heart Failure"]

healthy_or_not = df["deaths"].value_counts().tolist()
values = [healthy_or_not[0], healthy_or_not[1]]

fig = px.pie(
    values=df["deaths"].value_counts(),
    names=labels,
    width=700,
    height=400,
    color_discrete_sequence=["skyblue", "black"],
    title="Healthy vs Heart Failure",
)
fig.show()


# visualizing the age using distplot graph

sns.distplot(df.age)


# counting the number of
sns.countplot(x="diabetes", data=df)
plt.title("Count Plot of Diabetes")


# counting platelets

sns.scatterplot(x="age", y="platelets", data=df)
plt.title("Scatter Plot of Age vs Platelets")
plt.xlabel("Age")
plt.ylabel("Platelets")
plt.show()


# counting male and female
sns.countplot(x="sex", data=df)
plt.title("Count Plot of Male and Female")


# Data transformation with the original data

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


X

y


# training and testing out data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45)


X_train


y_train


# applying logistic Regression on the data

classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# performace before removing the outliers

y_pred = classifier.predict(X_test)
log = accuracy_score(y_test, y_pred) * 100
accuracy = accuracy_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred) * 100
f1Score = f1_score(y_test, y_pred) * 100

print("Accuracy Score:", accuracy, "%")
print("Recall Score:", recall, "%")
print("Precision Score:", precision, "%")
print("F1 Score:", f1Score, "%")
pd.crosstab(y_pred, y_test)


# Checking and Removing some outliers

# Drop the non-numeric column 'deaths' for outlier detection

numeric_cols = df.drop("deaths", axis=1)

# Calculate the IQR for each column

Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1

# Identify outliers

outliers = ((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(
    axis=1
)

# Display the rows containing outliers

print(df[outliers])


# Removing the outliers
dfno_out = df[~outliers]


# data after removing outliers

dfno_out


# data transformation after removing the outliers

X1 = dfno_out.iloc[:, :-1].values
y1 = dfno_out.iloc[:, -1].values
sc_X1 = StandardScaler()
X1 = sc_X1.fit_transform(X1)


X1


y1


# train and test
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.45)


X1_train


y1_train


# logistic regression on the new data

classifier1 = LogisticRegression()
classifier1.fit(X1_train, y1_train)


# displaying the performace after cleaning the data

y1_pred = classifier.predict(X1_test)
log = accuracy_score(y1_test, y1_pred) * 100
accuracy = accuracy_score(y1_test, y1_pred) * 100
recall = recall_score(y1_test, y1_pred) * 100
precision = precision_score(y1_test, y1_pred) * 100
f1Score = f1_score(y1_test, y1_pred) * 100

print("Accuracy Score:", accuracy, "%")
print("Recall Score:", recall, "%")
print("Precision Score:", precision, "%")
print("F1 Score:", f1Score, "%")
pd.crosstab(y1_pred, y1_test)


# pie chart representation for performance Analysis

scores = {
    "Accuracy Score": accuracy,
    "Recall Score": recall,
    "Precision Score": precision,
    "F1 Score": f1Score,
}

labels = scores.keys()
sizes = scores.values()

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
ax.axis("equal")

plt.title("Performance Metrics")
plt.show()


# performance analysis through AUC-ROC curve

y_pred_prob = classifier.predict_proba(X_test)[:, 1]

# AUC-ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(8, 8))
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=2,
    label="ROC curve (area = {:.2f})".format(roc_auc),
)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()
