import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
df = pd.read_csv('dataset.csv')
# Check column names to ensure 'result' exists
print("Columns in dataset:", df.columns)

# Remove any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Print column names again after stripping spaces
print("Updated Columns:", df.columns)


# Features and Target (Updating 'result' instead of 'match_winner')
x = df.drop('results', axis=1)  # Features
y = df['results']               # Target variable

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Preprocessing pipeline (One-Hot Encoding for categorical features)
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')


# Random Forest model
pipe = Pipeline([
    ('step1', trf),
    ('step2', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipe.fit(x_train, y_train)

# Save the model
joblib.dump(pipe, 'ra_pipe.joblib')

# Print model accuracy
y_pred = pipe.predict(x_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
