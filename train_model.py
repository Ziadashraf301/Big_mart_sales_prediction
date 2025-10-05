import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from datetime import datetime
import os

# --- Training pipeline with timestamp and metric logging ---
def train_save_rf_pipeline(
    df, 
    cat_vars, 
    excluded_features, 
    top_features, 
    target_col='Item_Outlet_Sales', 
    model_path='Models/',
    log_path='Models/training_log.csv'
):
    # Split features and target
    X = df.drop(excluded_features, axis=1)
    y = df[target_col]

    # --- One-Hot Encoding ---
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoder.fit(X[cat_vars])
    joblib.dump(encoder, model_path + "encoder.joblib")
    print(f"✅ OneHotEncoder saved as '{model_path}encoder.joblib'")

    encoded_df = pd.DataFrame(
        encoder.transform(X[cat_vars]),
        columns=encoder.get_feature_names_out(cat_vars),
        index=X.index
    )
    X = X.drop(cat_vars, axis=1)
    X_encoded = pd.concat([X, encoded_df], axis=1)
    print(f"Encoded shape: {X_encoded.shape}")

    # --- Scaling ---
    scaler = StandardScaler()
    scaler.fit(X_encoded)
    joblib.dump(scaler, model_path + "scaler.joblib")
    print(f"✅ Scaler saved as '{model_path}scaler.joblib'")

    X_scaled = pd.DataFrame(
        scaler.transform(X_encoded),
        columns=X_encoded.columns,
        index=X_encoded.index
    )
    print(f"Scaled shape: {X_scaled.shape}")

    # --- Select top features ---
    X_final = X_scaled[top_features]

    # --- Train Random Forest ---
    rf_model = RandomForestRegressor(
        n_estimators=1000,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_final, y)
    joblib.dump(rf_model, model_path + "RF_model.joblib")
    print(f"✅ Random Forest model saved as '{model_path}RF_model.joblib'")

    # --- Compute MAE on training data for monitoring ---
    y_pred_train = rf_model.predict(X_final)
    train_mae = mean_absolute_error(y, y_pred_train)
    print(f"📊 Training MAE: {train_mae:.2f}")

    # --- Log training timestamp and MAE ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = pd.DataFrame({
        'timestamp': [timestamp],
        'model': ['RandomForest'],
        'train_MAE': [train_mae]
    })

    if os.path.exists(log_path):
        log_entry.to_csv(log_path, mode='a', index=False, header=False)
    else:
        log_entry.to_csv(log_path, index=False)
    print(f"📝 Training log updated at '{log_path}'")

    return rf_model, encoder, scaler


# --- Demo Training ---
if __name__ == "__main__":
    cat_vars = ['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Type']
    excluded_features = ['Item_Identifier', 'Item_Visibility', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Item_Outlet_Sales']
    top_features = [
        'Item_MRP',
        'Outlet_Type_Supermarket Type1',
        'Outlet_Type_Supermarket Type3',
        'Item_Fat_Content_Regular',
        'Item_Weight',
        'Outlet_Location_Type_Tier 2',
        'Item_Type_Dairy',
        'Outlet_Type_Supermarket Type2',
        'Item_Type_Household',
        'Item_Type_Frozen Foods',
        'Item_Type_Soft Drinks',
        'Item_Type_Snack Foods',
        'Outlet_Location_Type_Tier 3'
    ]

    rf_model, encoder, scaler = train_save_rf_pipeline(
        df=pd.read_csv('transformed_data/BigMart_transformed_data.csv'),
        cat_vars=cat_vars,
        excluded_features=excluded_features,
        top_features=top_features,
        target_col='Item_Outlet_Sales',
        model_path='Models/'
    )
