import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json

def train_model():
    # Load dataset
    df = pd.read_csv("crop_yield_dataset.csv")
    
    # One-hot encode categorical features ("State" and "CropType")
    df_encoded = pd.get_dummies(df, columns=["State", "CropType"])
    
    # Define features and target
    X = df_encoded.drop(["Yield(kg_ha)", "Latitude", "Longitude"], axis=1)
    y = df_encoded["Yield(kg_ha)"]
    
    # Save the columns (features) order for later use in prediction
    model_columns = list(X.columns)
    with open("model_columns.json", "w") as f:
        json.dump(model_columns, f)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation R^2: {r2:.2f}")
    
    # Save the trained model
    joblib.dump(model, "crop_yield_model.pkl")
    print("Model saved to crop_yield_model.pkl")

if __name__ == "__main__":
    train_model()
