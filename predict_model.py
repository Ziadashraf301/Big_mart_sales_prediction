import pandas as pd
import joblib
import numpy as np


# --- Load models for inference ---
def load_rf_pipeline(model_path='Models/'):
    encoder = joblib.load(model_path + "encoder.joblib")
    scaler = joblib.load(model_path + "scaler.joblib")
    rf_model = joblib.load(model_path + "RF_model.joblib")
    print("✅ All models loaded successfully")
    return rf_model, encoder, scaler


# --- Forecasting function ---
def forecast_sales(new_data, excluded_features, cat_vars, top_features, encoder, scaler, model):
 
    new_data = new_data.drop(excluded_features, axis=1)

    encoded_df = pd.DataFrame(
        encoder.transform(new_data[cat_vars]),
        columns=encoder.get_feature_names_out(cat_vars),
        index=new_data.index
    )
    X = new_data.drop(cat_vars, axis=1)
    X_encoded = pd.concat([X, encoded_df], axis=1)
    X_scaled = pd.DataFrame(
        scaler.transform(X_encoded),
        columns=X_encoded.columns,
        index=X_encoded.index
    )
    X_final = X_scaled[top_features]
    predictions = model.predict(X_final)
    return pd.Series(predictions, index=new_data.index, name='Predicted_Sales')


# --- Demo Prediction ---
if __name__ == "__main__":
    
    cat_vars = ['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Type']
        
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

    excluded_features = ['Item_Identifier', 'Item_Visibility', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Item_Outlet_Sales']

    # Load saved models
    rf_model, encoder, scaler = load_rf_pipeline(model_path='Models/')

    # Create synthetic demo data
    demo_data = pd.read_csv('transformed_data/BigMart_transformed_data.csv')

    # Make predictions
    predicted_sales = forecast_sales(demo_data, excluded_features, cat_vars, top_features, encoder, scaler, rf_model)

    print("\n🎯 Demo Forecasted Sales:")
    print(predicted_sales)
