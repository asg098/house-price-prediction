import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
y = train_data.SalePrice
X = train_data.drop(['SalePrice'], axis=1)
object_cols = [col for col in X.columns if X[col].dtype == "object"]

low_cardinality_cols = [col for col in object_cols if X[col].nunique() < 10]
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

selected_cols = low_cardinality_cols + numerical_cols
X = X[selected_cols].copy()

X = pd.get_dummies(X)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=0)

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1)
model.fit(X_train, y_train, 
          early_stopping_rounds=5, 
          eval_set=[(X_valid, y_valid)], 
          verbose=False)

preds = model.predict(X_valid)
mae = mean_absolute_error(y_valid, preds)
print("Validation MAE:", mae)

test_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
X_test = test_data[selected_cols].copy()
X_test = pd.get_dummies(X_test)

X_test = X_test.reindex(columns=X.columns, fill_value=0)


test_preds = model.predict(X_test)

submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})

submission.to_csv('submission.csv', index=False)
