import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
import warnings
import joblib
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore")

# Step 1: Data Setup
data = pd.read_csv('data/Telco-Customer-Churn.csv')

# Display basic information
print(data.head())
print(data.info())
print(data.describe())

# Step 2: Data Preprocessing
# Fill missing values using forward fill
data.ffill(inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
categorical_columns = ['gender', 'Partner', 'Dependents']
for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])

# Ensure 'TotalCharges' is numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Fill missing numeric values with 0
numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
data[numeric_columns] = data[numeric_columns].fillna(0)

# Scale numerical features
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Step 3: Exploratory Data Analysis (EDA)
# Visualize the Churn Distribution
sns.countplot(data=data, x='Churn')
plt.title("Churn Distribution")
#plt.show()
plt.tight_layout()

# Correlation Heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
#plt.show()
plt.tight_layout()

# Step 4: Modeling
# Define features and target
X = data.drop(['Churn', 'customerID'], axis=1, errors='ignore')
y = encoder.fit_transform(data['Churn'])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Make predictions and probabilities
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the Model
print("Classification Report:")
print(classification_report(y_test, y_pred))

roc_score = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_score)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
#plt.show()
plt.tight_layout()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_score:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Random guess line
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
#plt.show()
plt.tight_layout()

# Step 5: Feature Importance
importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Visualize top 20 features
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
#plt.show()
plt.tight_layout()

# Step 6: Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_resampled, y_train_resampled)

# Train the best model
best_model = grid_search.best_estimator_
best_model.fit(X_train_resampled, y_train_resampled)

# Evaluate the best model
y_pred_tuned = best_model.predict(X_test)
y_prob_tuned = best_model.predict_proba(X_test)[:, 1]

print("Classification Report (Tuned Model):")
print(classification_report(y_test, y_pred_tuned))
print("ROC-AUC Score (Tuned Model):", roc_auc_score(y_test, y_prob_tuned))

# Step 7: Deployment with Streamlit
joblib.dump(best_model, 'churn_model.pkl')

st.title("Customer Churn Prediction")

# Input fields for customer details
tenure = st.number_input("Tenure", min_value=0, max_value=100)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict Churn"):
    model = joblib.load('churn_model.pkl')
    input_data = pd.DataFrame([[tenure, monthly_charges, total_charges]], 
                              columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
    prediction = model.predict(input_data)
    churn_probability = model.predict_proba(input_data)[0][1]
    st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
    st.write(f"Churn Probability: {churn_probability:.2f}")
