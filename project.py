import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

 #clean the dataset       
data = pd.read_csv("loan_data.csv")
print("First Five Rows: ",data.head())
print("Info: ",data.info())
print(data.describe())
print(data.shape)

data_cleaned = data.copy()

remove_duplicates = data_cleaned.drop_duplicates(inplace=True)
print("Removed Duplicates")
print(data.shape)

missing_values = data.isnull().sum()
print("Missing:\n",missing_values)


print(data.columns)

# numeric_columns = ['person_age','person_income','loan_amnt','loan_int_rate', 'loan_percent_income','credit_score']
# for col in numeric_columns:
#     plt.figure(figsize=(6,4))
#     sns.histplot(data[col],kde=True,bins=20)
#     plt.title(f"Histogram of {col}")
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
#     plt.show() 

# plt.figure(figsize=(8,6))
# sns.heatmap(data.corr(numeric_only=True),annot=True)
# plt.show()
# Converting string into number
numerical = LabelEncoder()

for col in data.select_dtypes(include='object'):
    data[col] = numerical.fit_transform(data[col])

# Split the data
X = data.drop("loan_status", axis=1)
y = data["loan_status"] 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check Target Balance

# print(y_train.value_counts())
# print(y_test.value_counts())

# Training the model
# print("-----------------------------------------------------------------------------------------------------------")
# print("Using LogisticRegression")
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# print(model)
# print("Model trained successfully")

# # Scale the data
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)  
# X_test = scaler.transform(X_test)  
# print(X_train)
# print(X_test)
# print("Scaling has done!")

# # Prediction
# y_pred = model.predict(X_test)
# print(y_pred)

# # Accuracy
# print("Accuracy: ", accuracy_score(y_test, y_pred))

# # confusion matrix
# print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

# # # classification report
# print("Classification Report:\n",classification_report(y_test, y_pred))

print("------------------------------------------------------------------------------------------------------------")
print("Using RandomForest")

# Training the Model
model1 = RandomForestClassifier()
model1.fit(X_train, y_train)

print(model1)
print("Model trained successfully")

# Prediction
y_pred_m1 = model1.predict(X_test)
print(y_pred_m1)

# Accuracy
print("Accuracy: ", accuracy_score(y_test, y_pred_m1))

# confusion matrix
print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred_m1))

# # classification report
print("Classification Report:\n",classification_report(y_test, y_pred_m1))


# print("------------------------------------------------------------------------------------------------------------")
# print("Using DecisionTreeClassifier")

# # Training the Model
# model2 = DecisionTreeClassifier()
# model2.fit(X_train, y_train)

# print(model2)
# print("Model trained successfully")

# # Prediction
# y_pred_m2 = model2.predict(X_test)
# print(y_pred_m2)

# # Accuracy
# print("Accuracy: ", accuracy_score(y_test, y_pred_m2))

# # confusion matrix
# print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred_m2))

# # # classification report
# print("Classification Report:\n",classification_report(y_test, y_pred_m2))


# Final Model means improving the performance, accuracy

# Tune Model

# from sklearn.model_selection import GridSearchCV                  # GridSearchCV -> tool to find best parameter

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }

# grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)     # Split the data into 5 parts in which 4 for training and 1 for testing

# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }

# grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, verbose=2,n_jobs=-1)
# grid.fit(X_train, y_train)

# print("Training started...")
# grid.fit(X_train, y_train)
# print("Training finished!")

# print("best parameters:\n",grid.best_params_)


# Feature Selection   ->   reduces noise
importance = model1.feature_importances_
features = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print(features)

# Removes less useful features

important_features = X_train.columns[importance > 0.02]
print(important_features)

X_train = X_train[important_features]
X_test = X_test[important_features]