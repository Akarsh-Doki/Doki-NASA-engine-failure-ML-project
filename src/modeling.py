import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint
import numpy as np

def train_xgboost(X_train, y_train, n_iter: int = 100):
# Train XGBoost with RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(50, 200),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': uniform(0, 0.5),
        'reg_lambda': uniform(0, 0.5)
    }

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=3, scoring="neg_mean_squared_error", random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def evaluate_model(model, X, y):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    return mse, rmse, preds