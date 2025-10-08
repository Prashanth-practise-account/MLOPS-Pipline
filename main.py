# data collection

import pandas as pd
data = pd.read_csv("dataset.csv")

#  data preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

# remove the duplicates and fill the empty value with starting number
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].mean())


data.drop_duplicates(inplace=True)

# -------------------
# Encode categorical columns
# -------------------
text_column = 'description'  # keep text column separate
for col1 in data.columns:
    if data[col1].dtype == "object" and col1!=text_column:
        le = LabelEncoder()
        data[col1] = le.fit_transform(data[col1])

# Text vectorization for 'description' only
vectorizer = CountVectorizer(max_features=50)
text_features = vectorizer.fit_transform(data[text_column]).toarray()
text_df = pd.DataFrame(text_features,columns=vectorizer.get_feature_names_out())
data = pd.concat([data.reset_index(drop=True), text_df.reset_index(drop=True)], axis=1)
data.drop(columns=['description'], inplace=True)

# -------------------
# Step 7: Feature Scaling for better ml models  perform better when features are scaled
# -------------------
scaler = StandardScaler()
numerical_cols = ['age', 'salary']  # scale numerical features
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Model Development & Model Evaluation
#  choose the model and train and evaluation

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X = data.drop(columns=['target'])
y=data['target']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 5: Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# Step 6: Train the model
rf_model.fit(X_train, y_train)
# Step 7: Validate the model
y_val_pred = rf_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Step 8: Test the model
y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

# model deployment

import joblib
# Save model and preprocessors
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(le, 'le_city.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(scaler, 'scaler.pkl')

# --------------------------
# Step 5: Predict New Sample
# --------------------------
# Example: new_data with target column for retraining
new_data = pd.DataFrame({
    'age': [29],
    'salary': [55000],
    'city': ['Bangalore'],
    'description': ['Enjoys outdoor activities'],
    'target': [1]  # <-- add this column for retraining
})

le_city = joblib.load('le_city.pkl')
new_data['city'] = le_city.transform(new_data['city'])
# Vectorize 'description' using saved vectorizer
vectorizer = joblib.load('vectorizer.pkl')
new_text_features = vectorizer.transform(new_data['description']).toarray()
text_df = pd.DataFrame(new_text_features, columns=vectorizer.get_feature_names_out())
new_data = pd.concat([new_data.reset_index(drop=True), text_df.reset_index(drop=True)], axis=1)
new_data.drop(columns=['description'], inplace=True)

# Scale numerical features using saved scaler
scaler = joblib.load('scaler.pkl')
new_data[['age', 'salary']] = scaler.transform(new_data[['age', 'salary']])

# Ensure column order matches training data
X_columns = X.columns
new_data = new_data[X_columns]

# Load model and predict
rf_model = joblib.load('rf_model.pkl')
prediction = rf_model.predict(new_data)
print("Predicted target:", prediction[0])

# Model Monitoring
import numpy as np

# Save training feature statistics for drift monitoring (do once during training)
train_means = X.mean()
train_stds = X.std()
joblib.dump(train_means, 'train_means.pkl')
joblib.dump(train_stds, 'train_stds.pkl')

# Example: New batch monitoring
new_means = new_data.mean()
new_stds = new_data.std()

# Calculate drift as % change
drift = np.abs(new_means - train_means) / (train_stds + 1e-5)
drift_features = drift[drift > 0.1]  # threshold >10%
print("\nFeatures with significant drift:\n", drift_features)

# Optional: If you have true labels for new data
true_labels = [1]  # replace with actual labels
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(true_labels, prediction)
print("Accuracy on new data:", accuracy)
print(classification_report(true_labels, prediction))

# Model Maintenance & Retraining

# Retraining condition
RETRAIN_THRESHOLD = 0.8
if accuracy < RETRAIN_THRESHOLD or len(drift_features) > 0:
    print("Retraining model...")

    # Combine old training data + new batch with target
    updated_data = pd.concat([data, new_data.reset_index(drop=True)], axis=0)

    # Ensure no NaN in target
    updated_data = updated_data.dropna(subset=['target'])

    X_updated = updated_data.drop(columns=['target'])
    y_updated = updated_data['target']

    # Retrain Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_updated, y_updated)

    # Save updated model
    joblib.dump(rf_model, 'rf_model.pkl')
    print("Model retrained and saved!")
