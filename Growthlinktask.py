import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load Dataset
df = pd.read_csv("tested")

# Display basic info
print(df.head())
print(df.info())

# Handling Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)  # Drop unnecessary columns

# Encoding Categorical Variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Feature Selection
X = df.drop(columns=['Survived'])
y = df['Survived']

# Check correlation before dropping features
corr_matrix = X.corr().abs()
high_corr_features = [column for column in corr_matrix.columns if any(corr_matrix[column] > 0.95) and column not in ['Fare', 'Age']]
X.drop(columns=high_corr_features, inplace=True)

# Normalizing Numerical Features
scaler = StandardScaler()
num_cols = X.select_dtypes(include=[np.number]).columns
X[num_cols] = scaler.fit_transform(X[num_cols])

# Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Model Training
model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save Model
joblib.dump(model, "titanic_model.pkl")
