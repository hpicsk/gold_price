"""Train and evaluate gold price regression models."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


class GoldPriceModels:
    """Container for training and evaluating multiple regression models."""

    def __init__(self):
        self.models = {
            "OLS (Linear Regression)": LinearRegression(),
            "Ridge Regression": Ridge(alpha=10.0),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
            ),
        }
        self.scaler = StandardScaler()
        self.results = {}
        self.feature_names = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train all models on scaled features."""
        self.feature_names = list(X_train.columns)
        X_scaled = self.scaler.fit_transform(X_train)

        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_scaled, y_train)

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Generate predictions from all models."""
        X_scaled = self.scaler.transform(X)
        return {name: model.predict(X_scaled) for name, model in self.models.items()}

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluate all models and return metrics DataFrame."""
        predictions = self.predict(X_test)
        rows = []
        for name, y_pred in predictions.items():
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rows.append({"Model": name, "RMSE": round(rmse, 2), "MAE": round(mae, 2), "R²": round(r2, 4)})
            self.results[name] = {"y_pred": y_pred, "rmse": rmse, "mae": mae, "r2": r2}

        return pd.DataFrame(rows)

    def get_feature_importance(self) -> dict:
        """Get feature importance/coefficients from each model."""
        importance = {}

        # Linear model coefficients
        for name in ["OLS (Linear Regression)", "Ridge Regression"]:
            model = self.models[name]
            importance[name] = dict(zip(self.feature_names, model.coef_))

        # Gradient boosting feature importance
        gb = self.models["Gradient Boosting"]
        importance["Gradient Boosting"] = dict(zip(self.feature_names, gb.feature_importances_))

        return importance
