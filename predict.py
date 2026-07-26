import os
import sys
import pandas as pd
import joblib

def load_artifacts():
    model_filename = 'knn_model.joblib'
    scaler_filename = 'scaler.joblib'
    
    if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
        print("Error: Model or scaler not found. Please run train_knn.py first to train and save the model.")
        sys.exit(1)
        
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    return model, scaler

def predict_speed(features_dict):
    model, scaler = load_artifacts()
    
    # Expected feature order
    feature_order = [
        'Temperature_C', 'Humidity_percent', 'Gas_ppm', 'Pressure_hPa', 
        'Occupancy', 'Voltage_V', 'Current_A'
    ]
    
    # Ensure all features exist in inputs
    input_data = []
    for feature in feature_order:
        if feature not in features_dict:
            raise ValueError(f"Missing required feature: {feature}")
        input_data.append(features_dict[feature])
        
    # Reshape and scale input features
    input_df = pd.DataFrame([input_data], columns=feature_order)
    scaled_features = scaler.transform(input_df)
    
    # Predict
    predicted_speed = model.predict(scaled_features)[0]
    return predicted_speed

def main():
    print("KNN Fan Speed Prediction CLI")
    print("-" * 30)
    
    # Sample input values for manual testing/demonstration
    sample_input = {
        'Temperature_C': 33.5,
        'Humidity_percent': 31.0,
        'Gas_ppm': 120.0,
        'Pressure_hPa': 1013.25,
        'Occupancy': 1.0,
        'Voltage_V': 12.0,
        'Current_A': 2.87
    }
    
    print("Testing with sample input values:")
    for k, v in sample_input.items():
        print(f"  {k}: {v}")
        
    try:
        predicted = predict_speed(sample_input)
        print(f"\nPredicted Fan Speed (MotorSpeed_RPM): {predicted:.2f} RPM")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
