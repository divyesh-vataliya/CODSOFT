# Credit Card Fraud Detection Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn import tree

import warnings
warnings.filterwarnings("ignore")


print("Loading the dataset...")
df = pd.read_csv(r'C:\vs\CODSOFT\Credit Card Fraud Detection\creditcard.csv')


print("\nFirst 5 rows:")
print(df.head())

print("\nData Info:")
print(df.info())


print("\nClass Distribution:")
print(df['Class'].value_counts())

# Normalize 'Amount' and 'Time'
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

# Reduce dataset size to speed up training (optional, for testing/demo only)
df = df.sample(frac=0.1, random_state=1)  # Use only 10% of the dataset for faster SVM

# Visualizations after data cleaning
plt.figure(figsize=(8, 5))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['Time'], bins=50, kde=True)
plt.title('Distribution of Transaction Times')
plt.xlabel('Time (scaled)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap of Features')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Class (0: Not Fraud, 1: Fraud)')
plt.ylabel('Count')
plt.show()


X = df.drop('Class', axis=1)
y = df['Class']

# Handle class imbalance using SMOTE
print("\nApplying SMOTE for class imbalance...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=1)

# Train and evaluate models

# Logistic Regression
print("\nTraining Logistic Regression model...")
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print("\nLogistic Regression Evaluation:")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Decision Tree
print("\nTraining Decision Tree Classifier model...")
dtree = DecisionTreeClassifier(random_state=1)
dtree.fit(X_train, y_train)
y_pred_dt = dtree.predict(X_test)
print("\nDecision Tree Classifier Evaluation:")
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Decision Tree Visualization
plt.figure(figsize=(20, 25))
tree.plot_tree(dtree,
               feature_names=X.columns,
               class_names=['Not Fraud', 'Fraud'],
               rounded=True,
               filled=True,
               proportion=True)
plt.title("Decision Tree Structure")
plt.show()

# K-Nearest Neighbors
print("\nTraining K-Nearest Neighbors model...")
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print("\nK-Nearest Neighbors Evaluation:")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# Support Vector Machine
print("\nTraining Support Vector Machine model (linear kernel)...")
svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=2)
svm_model.fit(X_train[:5000], y_train[:5000])  # Train on a smaller subset to avoid timeout
y_pred_svm = svm_model.predict(X_test)
y_pred_svm_proba = svm_model.predict_proba(X_test)[:, 1]

print("\nSupport Vector Machine Evaluation:")
print("Accuracy SVM:", accuracy_score(y_test, y_pred_svm))
print("Precision SVM:", precision_score(y_test, y_pred_svm))
print("Recall SVM:", recall_score(y_test, y_pred_svm))
print("F1 Score SVM:", f1_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Confusion Matrix Visualization for SVM
matrix_svm = confusion_matrix(y_test, y_pred_svm)
cm_svm = pd.DataFrame(matrix_svm, index=['Not Fraud', 'Fraud'], columns=['Not Fraud', 'Fraud'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm_svm, annot=True, cbar=None, cmap="Blues", fmt='g')
plt.title("Confusion Matrix - SVM")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.tight_layout()
plt.show()

# ROC Curve for SVM
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm_proba)
roc_auc_svm = auc(fpr_svm, tpr_svm)
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label='SVM ROC curve (area = {:.2f})'.format(roc_auc_svm))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - SVM')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve for SVM
precision_svm, recall_svm, _ = precision_recall_curve(y_test, y_pred_svm_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall_svm, precision_svm, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - SVM')
plt.grid()
plt.show()

# Visualize class balance after SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled)
plt.title("Class Distribution After SMOTE")
plt.show()



