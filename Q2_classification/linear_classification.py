import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# Load the training dataset
train_data = pd.read_csv("../data/classification/train.csv")

# Load the testing dataset
test_data = pd.read_csv("../data/classification/test.csv")

# Drop rows with missing values in both datasets
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# Concatenate training and testing datasets for one-hot encoding
combined_data = pd.concat([train_data, test_data], axis=0)

# One-hot encode categorical variables in combined dataset
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
combined_data_encoded = pd.get_dummies(combined_data, columns=categorical_cols)

# Split combined dataset back into training and testing datasets
train_data_encoded = combined_data_encoded[:len(train_data)]
test_data_encoded = combined_data_encoded[len(train_data):]

# Split features and target variable for both datasets
X_train = train_data_encoded.drop('income', axis=1)
y_train = train_data_encoded['income']
X_test = test_data_encoded.drop('income', axis=1)
y_test = test_data_encoded['income']

# Scale numerical features for both datasets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Inference on the first test sample
first_test_sample = X_test_scaled[:1]
first_test_prediction = model.predict(first_test_sample)
true_label = y_test.iloc[0]
print("Inference on the first test sample:")
print("True Label:", true_label)
print("Predicted Label:", first_test_prediction)
