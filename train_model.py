import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/Churn_Modelling.csv")

# Drop unnecessary columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Feature Engineering
df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
df['ProductDensity'] = df['NumOfProducts'] / (df['Tenure'] + 1)
df['AgeTenure'] = df['Age'] * df['Tenure']

# Features & Target
X = df.drop('Exited', axis=1)
y = df['Exited']
pickle.dump(X.columns.tolist(), open("features.pkl", "wb"))
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ model.pkl and scaler.pkl created!")
# Save feature names


