# Import libraries
import pandas as pd
import5 numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv('churn_data.csv')  # <-- Replace with your file

# View basic info
print(data.head())

# Encode categorical variables
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# Feature scaling
scaler = StandardScaler()
features = data.drop('Churn', axis=1)
target = data['Churn']
features_scaled = scaler.fit_transform(features)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Model building
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_
feat_importance = pd.Series(importances, index=features.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.show()
