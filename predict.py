import os
import sys
import argparse
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
    parser = argparse.ArgumentParser(description="Predict fan speed using trained KNN model.")
    parser.add_argument('--temp', type=float, help='Temperature in Celsius')
    parser.add_argument('--humidity', type=float, help='Humidity percent')
    parser.add_argument('--gas', type=float, help='Gas concentration in ppm')
    parser.add_argument('--pressure', type=float, help='Pressure in hPa')
    parser.add_argument('--occupancy', type=float, help='Occupancy (0 or 1)')
    parser.add_argument('--voltage', type=float, help='Voltage in V')
    parser.add_argument('--current', type=float, help='Current in A')
    parser.add_argument('--interactive', action='store_true', help='Enter inputs interactively')
    
    args = parser.parse_args()
    
    # Check if any CLI arguments were provided
    cli_args_provided = any(val is not None for val in [
        args.temp, args.humidity, args.gas, args.pressure, 
        args.occupancy, args.voltage, args.current
    ])
    
    inputs = {}
    if cli_args_provided:
        # Check all are provided
        required = {
            'Temperature_C': args.temp,
            'Humidity_percent': args.humidity,
            'Gas_ppm': args.gas,
            'Pressure_hPa': args.pressure,
            'Occupancy': args.occupancy,
            'Voltage_V': args.voltage,
            'Current_A': args.current
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            print(f"Error: Missing required CLI arguments: {missing}")
            print("Use: python predict.py --temp 30.5 --humidity 60 --gas 100 --pressure 1013 --occupancy 1 --voltage 12 --current 2.5")
            sys.exit(1)
        inputs = required
    elif args.interactive:
        print("Enter sensor readings interactively:")
        try:
            inputs['Temperature_C'] = float(input("Temperature (C): "))
            inputs['Humidity_percent'] = float(input("Humidity (%): "))
            inputs['Gas_ppm'] = float(input("Gas (ppm): "))
            inputs['Pressure_hPa'] = float(input("Pressure (hPa): "))
            inputs['Occupancy'] = float(input("Occupancy (0 or 1): "))
            inputs['Voltage_V'] = float(input("Voltage (V): "))
            inputs['Current_A'] = float(input("Current (A): "))
        except ValueError:
            print("Error: Invalid numerical input.")
            sys.exit(1)
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            sys.exit(0)
    else:
        print("No custom inputs provided. Running with sample test data...")
        print("Hint: Run with '--interactive' or use command-line arguments to test custom values.")
        inputs = {
            'Temperature_C': 33.5,
            'Humidity_percent': 31.0,
            'Gas_ppm': 120.0,
            'Pressure_hPa': 1013.25,
            'Occupancy': 1.0,
            'Voltage_V': 12.0,
            'Current_A': 2.87
        }
        
    print("\nInput parameters:")
    for k, v in inputs.items():
        print(f"  {k}: {v}")
        
    try:
        predicted = predict_speed(inputs)
        print(f"\nPredicted Fan Speed (MotorSpeed_RPM): {predicted:.2f} RPM")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()

