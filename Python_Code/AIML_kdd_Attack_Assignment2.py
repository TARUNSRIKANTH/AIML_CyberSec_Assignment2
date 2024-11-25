import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np

# Load the dataset
data = pd.read_csv('C:/Users/Tarun/PycharmProjects/AIML_Assignment2/kddcup.data_10_percent.gz', header=None)

# Check the number of columns in the dataset
print(f"Number of columns in dataset: {data.shape[1]}")

# Rename columns for better understanding (based on KDD dataset documentation)
columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'attack_type']

# Adjust this line according to your data's actual number of columns
if len(columns) == data.shape[1]:
    data.columns = columns
else:
    print(f"Warning: Column length mismatch. Expected {data.shape[1]}, got {len(columns)}.")
    # If column length is incorrect, print out the actual column names for debugging
    print(f"Actual columns: {data.columns.tolist()}")

# Data Cleaning
data.drop_duplicates(inplace=True)

# Encoding categorical variables (features and target)
label_encoders = {}
categorical_columns = ['protocol_type', 'service', 'flag']  # List of categorical features
for column in categorical_columns:
    if column in data.columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    else:
        print(f"Warning: Column '{column}' not found in the dataset.")

# Encoding the target column if it is categorical
target_column = "dst_bytes"  # Update this line with actual target column name
# Check if the target column is categorical (and convert it if necessary)
if target_column in data.columns:
    if data[target_column].dtype == 'object':  # Check if target column is categorical (string)
        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column])
else:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

# Separating features and target variable
X = data.drop(target_column, axis=1)  # Features
y = data[target_column]  # Target

# Ensure X contains only numeric values
X = X.apply(pd.to_numeric, errors='coerce')  # This ensures all columns are numeric, convert non-numeric to NaN
X = X.fillna(0)  # Replace NaN values with 0 (or any other strategy)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualizing correlation matrix
# Select only numeric columns for correlation matrix
numeric_data = X.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 10))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Selecting top features based on ANOVA F-value
selector = SelectKBest(score_func=f_classif, k=10)  # Select top 10 features
X_new = selector.fit_transform(X_train, y_train)

# Initialize models
models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

results = {}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    results[model_name] = report['accuracy']

# Display results
for model_name, accuracy in results.items():
    print(f"{model_name} Accuracy: {accuracy:.2f}")
