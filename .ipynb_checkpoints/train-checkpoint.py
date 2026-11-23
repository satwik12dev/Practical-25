
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Download and load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

df = pd.read_csv(url, names=columns)

print("Dataset downloaded successfully!")
print("Dataset Shape:", df.shape)

# 2. Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Handle missing values
zero_columns = [
    'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI'
]

X[zero_columns] = X[zero_columns].replace(0, np.nan)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Create ML pipeline
model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# 6. Train model
model.fit(X_train, y_train)

print("Model trained successfully!")

# 7. Evaluate model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions) * 100, "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 8. Save model
os.makedirs('model', exist_ok=True)

joblib.dump(model, 'model/diabetes_model.pkl')

print("\nModel saved successfully!")
