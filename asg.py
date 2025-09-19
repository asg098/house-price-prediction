import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load training data
train_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
y = train_data.SalePrice
X = train_data.drop(['SalePrice'], axis=1)

# Identify categorical columns
object_cols = [col for col in X.columns if X[col].dtype == "object"]

# Separate low and high cardinality columns
low_cardinality_cols = [col for col in object_cols if X[col].nunique() < 10]
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

# Final columns to keep
selected_cols = low_cardinality_cols + numerical_cols
X = X[selected_cols].copy()

# One-hot encode low-cardinality categorical columns
X = pd.get_dummies(X)

# Split into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=0)

# Define and train model
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1)
model.fit(X_train, y_train, 
          early_stopping_rounds=5, 
          eval_set=[(X_valid, y_valid)], 
          verbose=False)

# Evaluate model
preds = model.predict(X_valid)
mae = mean_absolute_error(y_valid, preds)
print("Validation MAE:", mae)

# Load test data
test_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
X_test = test_data[selected_cols].copy()
X_test = pd.get_dummies(X_test)

# Align test data with training data
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Predict on test data
test_preds = model.predict(X_test)

# Save submission file
submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
submission.to_csv('submission.csv', index=False)