
# ml.py: Model training and prediction interface
import kagglehub
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Download and preprocess dataset, train model, save model
def train_and_save_model():
  path = kagglehub.dataset_download("ziya07/student-health-data")
  data = pd.read_csv(os.path.join(path, 'student_health_data.csv'))
  mask = pd.Series(True, index=data.index)
  for col in data.columns:
    if data[col].dtype == 'int64' or data[col].dtype == 'float64':
      Q1 = data[col].quantile(0.25)
      Q3 = data[col].quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      mask &= (data[col] >= lower_bound) & (data[col] <= upper_bound)
  data_cleaned = data[mask]
  data1 = data_cleaned.copy()
  label_encoder = LabelEncoder()
  for col in data1.columns:
    if data1[col].dtype == 'object':
      data1[col] = label_encoder.fit_transform(data1[col])
  y = data1.iloc[:, -1]
  X = data1.drop(columns=['Student_ID', 'Health_Risk_Level'])
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = RandomForestClassifier()
  model.fit(X_train, y_train)
  joblib.dump(model, 'random_forest_model.joblib')
  return model

# Load model (train if not exists)
def load_model():
  model_path = 'random_forest_model.joblib'
  if not os.path.exists(model_path):
    return train_and_save_model()
  return joblib.load(model_path)

# Prepare input for prediction
def prepare_input(user_input: dict):
  # Map categorical values to expected encoding
  # This should match the encoding used in training
  # For demo, use simple mappings
  gender_map = {"Male": 1, "Female": 0, "Other": 2, "Prefer not to say": 3}
  activity_map = {"High": 0, "Moderate": 1, "Low": 2, "None": 3}
  sleep_map = {"Good": 0, "Fair": 1, "Poor": 2, "Very Poor": 3}
  mood_map = {"Happy": 0, "Neutral": 1, "Anxious": 2, "Sad": 3, "Irritable": 4, "Other": 5}
  # Accepts already encoded dict from app.py, drops unused columns, ensures column order
  df = pd.DataFrame([user_input])
  for col in ['Student_ID', 'Health_Risk_Level']:
    if col in df.columns:
      df = df.drop(columns=[col])
  # Ensure columns match training order
  model = load_model()
  if hasattr(model, 'feature_names_in_'):
    expected_cols = list(model.feature_names_in_)
    df = df.reindex(columns=expected_cols)
  return df

# Predict function to be called from Streamlit
def predict_health_risk(user_input: dict):
  model = load_model()
  X_input = prepare_input(user_input)
  pred = model.predict(X_input)
  # For demo, map prediction to label
  label_map = {0: "Low", 1: "Moderate", 2: "High"}
  return label_map.get(int(pred[0]), "Unknown")