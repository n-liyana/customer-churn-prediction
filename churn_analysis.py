
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Step2: Data Setup
# Load the dataset
data=pd.read_csv('data/Telco-Customer-Churn.csv')

# Display basic information
print(data.head())
print(data.info())
print(data.describe())

# Step3: Data Prepocessing
# Check for missing values
print(data.isnull().sum())

# Fill missing values
#data.fillna(method='ffill', inplace=True)
data.ffill(inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
data['gender']=encoder.fit_transform(data['gender'])
data['Partner']=encoder.fit_transform(data['Partner'])
data['Dependents']=encoder.fit_transform(data['Dependents'])

#ensure contain non-numeric values
print(data[['tenure','MonthlyCharges','TotalCharges']].dtypes)

#convert to numeric
data['TotalCharges']=pd.to_numeric(data['TotalCharges'], errors='coerce')

#check missing values again after conversion
print(data[['tenure','MonthlyCharges','TotalCharges']].isnull().sum())

#fill or drop missing values
data[['tenure','MonthlyCharges','TotalCharges']]=data[['tenure','MonthlyCharges','TotalCharges']].fillna(0)

#Scaling numerical features
scaler = StandardScaler()
data[['tenure', 'MonthlyCharges', 'TotalCharges']]=scaler.fit_transform(
    data[['tenure','MonthlyCharges','TotalCharges']])

# Step4: Exploratory Data Analysis (EDA)
# Visualize the Churn Distribution
import matplotlib.pyplot as plt
import seaborn as sns

#check for non-numeric
print (data.dtypes)

#remove non-numeric columns (drop non-numeric)
numeric_data=data.select_dtypes(include=['number'])

#handle missing values (check missing values)
print(numeric_data.isnull().sum())

#drop or fill missing values
numeric_data=numeric_data.fillna(0) #replace NaN with 0

#compute correlation
correlation_matrix=numeric_data.corr()

sns.countplot(data['Churn'])
plt.title("Churn Distribution")
plt.savefig("Churn Distribution.png")
#plt.show()
plt.close()  # Close the plot to release resources

# Correlation Analysis
# Heatmap of correlations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("Correlation Matrix.png")
#plt.show()
plt.close()  # Close the plot to release resources

# Close the plot to prevent lingering resources
#plt.close()

# Step5: Modeling
# 1. Split the data
from sklearn.model_selection import train_test_split

#define features and target
X=data.drop('Churn', axis=1)
y=data['Churn']

#Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# check before training
print (X_train.dtypes)
print (y_train.dtypes)

# check for missing values - no Nan or missing values
print(X_train.isnull().sum())
print(y_train.isnull().sum())

# if have, fill them
X_train = X_train.fillna(0)
y_train = y_train.fillna(0)

# check the shape of X_train and y_train
print(X_train.shape)
print(y_train.shape)

# if dont match, debug the splitting step
from sklearn.model_selection import train_test_split

#define features and target
X=data.drop('Churn', axis=1)
y=data['Churn']

#Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#convert target variable to binary (if needed)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

#Additional checks (categorical columns in X_train)
X_train=pd.get_dummies(X_train,drop_first=True)
X_test=pd.get_dummies(X_test, drop_first=True)

# Align the columns of X_test with X_train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

#check features names
print("Training features:", X_train.columns)
print("Testing features:", X_test.columns)

#retry training model
# 2. Train a random forest model
from sklearn.ensemble import RandomForestClassifier

#Train the model
model=RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# 3. Evaluate the model
from sklearn.metrics import classification_report, roc_auc_score

# print evaluation metrics
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# Step6: Feature Importance
# Visualize feature importance
importance=model.feature_importances_

# verify lengths of importance and X.columns
print("Length of importance:", len(importance))
print("Number of features:", len(X.columns))

#Debug the Mismatch
#print("Training feature names:", X_train.columns)
#print("Model feature importance length:", len(importance))

#Confirm One-Hot Encoding or Similar Expansion
#print("Number of original features:", len(X.columns))  # Before preprocessing
#print("Number of features after preprocessing:", len(X_train.columns))  # After one-hot encoding

#Verify the Lengths Again
#print("Length of feature importances:", len(importance))
#print("Number of training features:", len(X_train.columns))

print("Number of training features:", len(X_train.columns))
print("Number of test features:", len(X_test.columns))
print("Training features:", X_train.columns)
print("Test features:", X_test.columns)



#Align Training and Testing Features
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Align the columns
X_train, X_test = X_train.align(X_test, join='inner', axis=1)

#Drop Unnecessary Columns
X_train = X_train.drop(['customerID'], axis=1, errors='ignore')
X_test = X_test.drop(['customerID'], axis=1, errors='ignore')

#Ensure X.columns Matches model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,    # Use X_train columns, not X.columns
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

#Ensure Consistent Preprocessing
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Align columns
X_train, X_test = X_train.align(X_test, join='inner', axis=1)

#Check for Extra or Missing Columns
print("Training features:", X_train.columns)
print("Test features:", X_test.columns)

# Fix the Mismatch
#importance_df=pd.DataFrame({
    #'Feature': X.columns[:len(importance)],  # Match the length of importance
    #'Importance': importance
    #}).sort_values(by='Importance', ascending=False)

#Check and Remove Non-Numeric Columns
X= data.drop(['Churn', 'customerID'],axis=1)

#Recalculate Feature Importance
model.fit(X_train, y_train)
importance = model.feature_importances_

#create dataframe for plotting
import pandas as pd

#combine feature importance and column names into a Dataframe
#importance_df=pd.DataFrame({
#    'Feature': X.columns,  
#    'Importance': importance
#}).sort_values(by='Importance', ascending=False)

#plot with Seaborn
import seaborn as sns
import matplotlib.pyplot as plt

top_features = importance_df.head(20)

#plot feature importance
sns.barplot(x='Importance', y='Feature', data=top_features, hue='Feature', palette='viridis', legend=False)
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("Top 20 Feature Importances.png")
#plt.show()
plt.close()  # Close the plot to release resources

# Step7: Hyperparameter tuning (Optional)
# Use GridSearchCV or RandomizedSearchCV to optimize the model
from sklearn.model_selection import GridSearchCV

param_grid={
    'n_estimators':[100,200,300],
    'max_depth':[5,10,15],
    'min_samples_split':[2,5,10],
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Best parameters
print(grid_search.best_params_)

from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

#---------------------------------------------------------------
# Evaluate with Performance Metrics
#make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class (churn)

#Generate a Classification Report:
print("Classification Report : ")
print (classification_report(y_test, y_pred))

#compute ROC-AUC score

roc_score = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_score)

#Generate the Matrix
cm = confusion_matrix(y_test, y_pred)

#Visualize the Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Confusion Matrix.png")
#plt.show()
plt.close()  # Close the plot to release resources

#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_score:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Random guess line
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("ROC Curve.png")
#plt.show()
plt.close()  # Close the plot to release resources

#Feature Importance Analysis
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importance_df.head(10))  # Top 10 important features

#Check Overfitting
# Training metrics
train_pred = model.predict(X_train)
train_prob = model.predict_proba(X_train)[:, 1]

print("Training Classification Report:")
print(classification_report(y_train, train_pred))
print("Training ROC-AUC Score:", roc_auc_score(y_train, train_prob))

# Compare with test metrics

#---------------------------------------------------------------------------
# 1. Improve Model Recall for Churn (Class 1)
# Oversampling (e.g., SMOTE) or Undersampling.

y_pred_adjusted = (y_prob > 0.4).astype(int)

#Adjusting Decision Threshold:
from sklearn.metrics import classification_report

# Adjust decision threshold
threshold = 0.4
y_pred_adjusted = (y_prob > threshold).astype(int)

# Evaluate adjusted predictions
print("Classification Report with Adjusted Threshold:")
print(classification_report(y_test, y_pred_adjusted))

#Applying SMOTE for Class Imbalance:
from imblearn.over_sampling import SMOTE

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train model with resampled data
model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_resampled = model.predict(X_test)
y_prob_resampled = model.predict_proba(X_test)[:, 1]

# Evaluate the model with the resampled training data
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("Classification Report (Resampled):")
print(classification_report(y_test, y_pred_resampled))

roc_score_resampled = roc_auc_score(y_test, y_prob_resampled)
print("ROC-AUC Score (Resampled):", roc_score_resampled)

# Confusion Matrix
cm_resampled = confusion_matrix(y_test, y_pred_resampled)

# Visualize Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(cm_resampled, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title("Confusion Matrix (Resampled)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Confusion Matrix (Resampled).png")
#plt.show()
plt.close()  # Close the plot to release resources

# 3. Optimize the Model (Hyperparameter Tuning)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_resampled, y_train_resampled)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Train with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train_resampled, y_train_resampled)

# Evaluate the tuned model
y_pred_tuned = best_model.predict(X_test)
y_prob_tuned = best_model.predict_proba(X_test)[:, 1]

print("Classification Report (Tuned Model):")
print(classification_report(y_test, y_pred_tuned))
print("ROC-AUC Score (Tuned Model):", roc_auc_score(y_test, y_prob_tuned))

# 4. Refine Feature Importance

# Feature importance from the tuned model
importance_tuned = best_model.feature_importances_

# Create a DataFrame for the top features
importance_df_tuned = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance_tuned
}).sort_values(by='Importance', ascending=False)

# Visualize the top 20 features
sns.barplot(x='Importance', y='Feature', data=importance_df_tuned.head(20), hue='Feature', palette='viridis', legend=False)
plt.title("Top 20 Feature Importances (Tuned Model)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("Top 20 Feature Importances (Tuned Model).png")
#plt.show()
plt.close()  # Close the plot to release resources

# 6. Deployment (Optional)
# Deployment with Streamlit
# Save the model and create a Streamlit app for predictions
import joblib
joblib.dump(best_model, 'churn_model.pkl')
# After grid search
joblib.dump(grid_search.best_estimator_, 'churn_model.pkl')

# Streamlit app
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
# In Streamlit app)
model = joblib.load('churn_model.pkl')

import os # Import the os module

output_dir = "outputs/"
os.makedirs(output_dir, exist_ok=True)   # Ensure the output directory exists

# Save churn distribution plot
sns.countplot(x='Churn', data=data)
plt.title("Churn Distribution")
plt.savefig(os.path.join(output_dir, "Churn Distribution.png"))
plt.close()

# Save Correlation Matrix plot
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig(os.path.join(output_dir, "Correlation Matrix.png"))
plt.close()

# Save Confusion Matrix plot
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(output_dir, "Confusion Matrix.png"))
plt.close()

# Save ROC Curve plot
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_score))
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, "ROC Curve.png"))
plt.close()

# Define the feature names used during training
# Make sure this matches the feature names from your training data after preprocessing
#feature_names = model.feature_importances_
feature_names = X_train.columns

st.title("Customer Churn Prediction")

# Add instructions
st.write("""
This app predicts the likelihood of a customer churning based on their details.
Please fill in the fields below and click **Predict Churn** to get the results.
""")

# Input fields for customer details
tenure = st.number_input("Tenure (Number of months the customer has stayed with the company)", min_value=0, max_value=100, value=24, step=1)
monthly_charges = st.number_input("Monthly Charges  (Average monthly charge in $)", min_value=0.0, value=65.0, step=0.01)
total_charges = st.number_input("Total Charges(Cumulative total charges in $)", min_value=0.0, value=1560.0, step=0.01)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No", "Yes"])

# Preprocess input data to match training data
def preprocess_input(tenure, monthly_charges, total_charges, contract, dependents, device_protection):
     # Create a DataFrame for user input
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Contract': [contract],
        'Dependents': [dependents],
        'DeviceProtection': [device_protection]
    })

    # Check for missing or invalid inputs
    input_data.fillna(0, inplace=True)  # Replace missing values with 0

    # Perform one-hot encoding
    input_data = pd.get_dummies(input_data)

    # Align with training features
    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    
    return input_data

# Predict churn
if st.button("Predict Churn"):
    try:
        model = joblib.load('churn_model.pkl')
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first.")
        st.stop()
    
    # Preprocess input data
    input_data = preprocess_input(tenure, monthly_charges, total_charges, contract, dependents, device_protection)
    if input_data.empty:
        st.error("Invalid input. Please ensure all fields are correctly filled.")
    else:
        try:
            # Predict churn and probability
            prediction = model.predict(input_data)
            churn_probability = model.predict_proba(input_data)[0][1]
            
            # Display Results
            st.write(f"**Churn Prediction:** {'Yes' if prediction[0] == 1 else 'No'}")
            st.write(f"**Churn Probability:** {churn_probability:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # All preprocessing steps here
    return X_train, X_test, y_train, y_test