import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_data

def run_hyperparameter_models():
    df = load_data()
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Ridge Regression": {
            "model": Ridge(),
            "params": {
                "alpha": [0.1, 1.0, 10.0]
            }
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5, 10]
            }
        }
    }

    for name, config in models.items():
        print(f"Running {name}...")
        grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        predictions = best_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"{name} Best Params: {grid.best_params_}")
        print(f"{name} -> MSE: {mse:.2f}, R²: {r2:.2f}\n")

if __name__ == "__main__":
    run_hyperparameter_models()